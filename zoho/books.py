"""Zoho Books API wrapper.

This module provides a production-ready, typed, and robust client for Zoho Books
API operations used by the project. It isolates Zoho-specific logic so the
rest of the application (evaluation and LLM wrappers) need not be changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

LOGGER = logging.getLogger("zoho.books")
LOGGER.addHandler(logging.NullHandler())


class ZohoError(Exception):
    """Base class for Zoho-related errors."""


class ZohoAuthError(ZohoError):
    """Raised for authentication / token related failures."""


class ZohoAPIError(ZohoError):
    """Raised when the Zoho API returns an unexpected response or status."""


class ZohoValidationError(ZohoError):
    """Raised when input validation fails before making an API request."""


@dataclass
class ZohoBooksClient:
    """Zoho Books API client.

    The client reads configuration from environment variables by default but
    can be instantiated directly for testing.
    """

    client_id: str
    client_secret: str
    refresh_token: str
    organization_id: str
    base_accounts_url: str = field(default_factory=lambda: os.getenv("ZOHO_BASE_URL", "https://www.zohoapis.in"))
    token_url: str = field(default_factory=lambda: os.getenv("ZOHO_TOKEN_URL", "https://accounts.zoho.in/oauth/v2/token"))
    timeout: int = field(default_factory=lambda: int(os.getenv("ZOHO_TIMEOUT", "30")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("ZOHO_MAX_RETRIES", "3")))
    backoff_factor: float = field(default_factory=lambda: float(os.getenv("ZOHO_BACKOFF_FACTOR", "0.5")))

    # In-memory token cache
    _access_token: Optional[str] = field(default=None, init=False)
    _access_token_expires_at: float = field(default=0.0, init=False)

    @classmethod
    def from_env(cls) -> "ZohoBooksClient":
        """Create a client from environment variables.

        Required env vars: `ZOHO_CLIENT_ID`, `ZOHO_CLIENT_SECRET`,
        `ZOHO_REFRESH_TOKEN`, `ZOHO_ORGANIZATION_ID`.
        """
        return cls(
            client_id=os.getenv("ZOHO_CLIENT_ID", ""),
            client_secret=os.getenv("ZOHO_CLIENT_SECRET", ""),
            refresh_token=os.getenv("ZOHO_REFRESH_TOKEN", ""),
            organization_id=os.getenv("ZOHO_ORGANIZATION_ID", ""),
        )

    def _debug_request(self, method: str, url: str, **kwargs: Any) -> None:
        """Print an informative console line about the outgoing request.

        Does NOT print secrets (tokens, client secret, refresh token).
        """
        masked_headers = dict(kwargs.get("headers") or {})
        if "Authorization" in masked_headers:
            masked_headers["Authorization"] = "<REDACTED>"

        LOGGER.info("Zoho request: %s %s", method.upper(), url)
        print(f"REQUEST -> {method.upper()} {url}")
        print(f"HEADERS -> {masked_headers}")
        if "json" in kwargs and kwargs["json"] is not None:
            try:
                pretty = json.dumps(kwargs["json"], ensure_ascii=False)
            except Exception:
                pretty = str(kwargs["json"])
            print(f"PAYLOAD -> {pretty}")

    def _request_with_retries(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        """Perform an HTTP request with retries and exponential backoff.

        Retries on 5xx and 429 responses and on network errors.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self._debug_request(method, url, **kwargs)
                # Use the convenience methods (get/post/put/delete) so tests that
                # patch `requests.get` / `requests.post` continue to work.
                method_name = method.lower()
                method_func = getattr(requests, method_name, None)
                if method_func is None:
                    # fallback to requests.request
                    resp = requests.request(method, url, timeout=self.timeout, **kwargs)
                else:
                    resp = method_func(url, timeout=self.timeout, **kwargs)
                LOGGER.debug("HTTP %s %s -> %s", method, url, resp.status_code)
                # Print a masked summary of the response body for debugging.
                try:
                    body = resp.json()
                    if isinstance(body, dict) and "access_token" in body:
                        body = dict(body)
                        body["access_token"] = "<REDACTED>"
                    print("RESPONSE ->", resp.status_code)
                    print("BODY ->", body)
                except Exception:
                    print("RESPONSE ->", resp.status_code)
                    print("BODY -> <non-json or unreadable>")
                if resp.status_code in (429,) or 500 <= resp.status_code < 600:
                    # transient server error, retry
                    last_exc = ZohoAPIError(f"Transient error {resp.status_code}: {resp.text}")
                    backoff = self.backoff_factor * (2 ** (attempt - 1))
                    LOGGER.warning("Transient error (attempt %s/%s): %s; retrying in %.2fs", attempt, self.max_retries, resp.status_code, backoff)
                    time.sleep(backoff)
                    continue
                return resp
            except requests.RequestException as exc:
                last_exc = exc
                backoff = self.backoff_factor * (2 ** (attempt - 1))
                LOGGER.warning("Network error (attempt %s/%s): %s; retrying in %.2fs", attempt, self.max_retries, exc, backoff)
                time.sleep(backoff)
                continue

        # exhausted retries
        raise ZohoAPIError("Request failed after retries") from last_exc

    def _fetch_access_token(self) -> Tuple[str, int]:
        """Exchange refresh token for an access token.

        Returns a tuple (access_token, expires_in_seconds).
        Raises `ZohoAuthError` on failure.
        """
        params = {
            "refresh_token": self.refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "refresh_token",
        }

        try:
            resp = requests.post(self.token_url, params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            LOGGER.exception("Token exchange failed")
            raise ZohoAuthError("Failed to reach token endpoint") from exc

        if resp.status_code >= 400:
            LOGGER.error("Token exchange failed: %s %s", resp.status_code, resp.text)
            raise ZohoAuthError(f"Token exchange failed: {resp.status_code}")

        data = resp.json()
        token = data.get("access_token")
        if not token:
            LOGGER.error("Token response missing access_token: %s", data)
            raise ZohoAuthError("Token response did not contain access_token")

        expires_in = int(data.get("expires_in", 3600))
        return token, expires_in

    def get_access_token(self) -> str:
        """Return a valid access token, using an in-memory cache until expiry.

        This method is safe to call repeatedly and will refresh the token
        automatically when expired.
        """
        now = time.time()
        # small safety margin of 30 seconds
        if self._access_token and now + 30 < self._access_token_expires_at:
            return self._access_token

        token, expires_in = self._fetch_access_token()
        self._access_token = token
        self._access_token_expires_at = now + int(expires_in)
        LOGGER.info("Fetched new access token, expires in %s seconds", expires_in)
        return token

    def _auth_headers(self) -> Dict[str, str]:
        """Return standard headers for authenticated requests.

        The Authorization header value is masked in logs and printouts.
        """
        token = self.get_access_token()
        return {"Authorization": f"Zoho-oauthtoken {token}", "Content-Type": "application/json"}

    def list_organizations(self) -> Dict[str, Any]:
        """List organizations available for the current credentials.

        Returns the parsed JSON response as a Python dictionary.
        """
        url = f"{self.base_accounts_url}/books/v3/organizations"
        resp = self._request_with_retries("get", url, headers=self._auth_headers())
        if resp.status_code != 200:
            raise ZohoAPIError(f"Failed to list organizations: {resp.status_code}")
        return resp.json()

    def list_chart_of_accounts(self) -> Dict[str, Any]:
        """Return the chart of accounts for the configured organization.

        The result is the parsed JSON response from Zoho Books.
        """
        url = f"{self.base_accounts_url}/books/v3/chartofaccounts?organization_id={self.organization_id}"
        resp = self._request_with_retries("get", url, headers=self._auth_headers())
        if resp.status_code != 200:
            raise ZohoAPIError(f"Failed to list chart of accounts: {resp.status_code}")
        return resp.json()

    # backward-compatible alias
    list_accounts = list_chart_of_accounts

    def list_contacts(self, contact_type: Optional[str] = None) -> Dict[str, Any]:
        """List contacts; optionally filter by `contact_type` (e.g., 'vendor')."""
        url = f"{self.base_accounts_url}/books/v3/contacts?organization_id={self.organization_id}"
        if contact_type:
            url += f"&contact_type={contact_type}"
        resp = self._request_with_retries("get", url, headers=self._auth_headers())
        if resp.status_code != 200:
            raise ZohoAPIError(f"Failed to list contacts: {resp.status_code}")
        return resp.json()

    def list_vendors(self) -> Dict[str, Any]:
        """List vendor contacts.

        Some Zoho APIs surface vendors via `contacts` with `contact_type=vendor`.
        """
        return self.list_contacts(contact_type="vendor")

    def list_taxes(self) -> Dict[str, Any]:
        """Return tax settings available in the organization."""
        url = f"{self.base_accounts_url}/books/v3/settings/taxes?organization_id={self.organization_id}"
        resp = self._request_with_retries("get", url, headers=self._auth_headers())
        if resp.status_code != 200:
            raise ZohoAPIError(f"Failed to list taxes: {resp.status_code}")
        return resp.json()

    def create_vendor_if_not_exists(self, vendor_name: str, **extra_fields: Any) -> Dict[str, Any]:
        """Ensure a vendor exists with the given name and return the vendor record.

        If a vendor with a matching `contact_name` exists, that record is returned.
        Otherwise a new vendor contact is created.
        """
        vendors = self.list_vendors()
        items = vendors.get("contacts") or vendors.get("vendors") or []
        for v in items:
            if v.get("contact_name", "").strip().lower() == vendor_name.strip().lower():
                LOGGER.info("Found existing vendor: %s", vendor_name)
                return v

        # create vendor
        payload: Dict[str, Any] = {"contact_name": vendor_name, "contact_type": "vendor"}
        payload.update(extra_fields)
        url = f"{self.base_accounts_url}/books/v3/contacts?organization_id={self.organization_id}"
        resp = self._request_with_retries("post", url, headers=self._auth_headers(), json=payload)
        if resp.status_code not in (200, 201):
            raise ZohoAPIError(f"Failed to create vendor: {resp.status_code} {resp.text}")
        data = resp.json()
        created = data.get("contact") or (data.get("contacts") and data.get("contacts")[0])
        if not created:
            raise ZohoAPIError("Unexpected vendor creation response")
        LOGGER.info("Created vendor: %s (id=%s)", vendor_name, created.get("contact_id"))
        return created

    def _choose_account_ids(self, bill: Dict[str, Any]) -> Tuple[str, str]:
        """Heuristically determine `account_id` and `paid_through_account_id`.

        The strategy is:
        - account_id: pick first account marked as 'Expense' or name contains 'expense'.
        - paid_through_account_id: prefer 'Bank' or 'Cash' accounts.
        Raises `ZohoAPIError` if suitable accounts cannot be found.
        """
        accounts_resp = self.list_chart_of_accounts()
        accounts = accounts_resp.get("chartofaccounts") or accounts_resp.get("accounts") or []

        account_id = ""
        paid_through = ""

        for a in accounts:
            name = (a.get("account_name") or a.get("name") or "").lower()
            acct_type = (a.get("account_type") or a.get("type") or "").lower()
            if not account_id and ("expense" in acct_type or "expense" in name or acct_type == "expense"):
                account_id = a.get("account_id") or a.get("account_code") or a.get("account_name")
            if not paid_through and ("bank" in acct_type or "cash" in acct_type or "bank" in name or "cash" in name):
                paid_through = a.get("account_id") or a.get("account_code") or a.get("account_name")

        if not account_id:
            # fallback: any non-liability account
            for a in accounts:
                acct_type = (a.get("account_type") or a.get("type") or "").lower()
                if acct_type and acct_type != "liability":
                    account_id = a.get("account_id") or a.get("account_code") or a.get("account_name")
                    break

        if not paid_through and accounts:
            # choose first account as last resort
            a = accounts[0]
            paid_through = a.get("account_id") or a.get("account_code") or a.get("account_name")

        if not account_id or not paid_through:
            raise ZohoAPIError("Unable to auto-select account or paid-through account; inspect chart of accounts")

        return str(account_id), str(paid_through)

    def _validate_bill(self, bill: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize incoming `bill` payload from the UI/LLM.

        Required: `vendor`, `amount`, `date`.
        Optional: `bill_no`, `currency`, `gst`.
        Returns normalized dict with typed `amount`.
        """
        missing: List[str] = []
        if not bill.get("vendor"):
            missing.append("vendor")
        if not bill.get("amount"):
            missing.append("amount")
        if not bill.get("date"):
            missing.append("date")
        if missing:
            raise ZohoValidationError(f"Missing required bill fields: {', '.join(missing)}")

        normalized: Dict[str, Any] = dict(bill)
        try:
            normalized["amount"] = float(bill["amount"])
        except Exception as exc:
            raise ZohoValidationError("`amount` must be numeric") from exc

        # basic date check (ISO-like)
        if not isinstance(normalized.get("date"), str) or len(normalized.get("date")) < 8:
            raise ZohoValidationError("`date` must be an ISO-like date string YYYY-MM-DD")

        return normalized

    def create_expense(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Create an expense in Zoho Books.

        Backwards-compatible signature supported:
        - `create_expense(account_id, paid_through_account_id, bill)` (legacy tests)
        - `create_expense(bill)` (preferred, automatic account selection)

        The method validates and normalizes `bill`, ensures the vendor exists,
        auto-selects accounts when not provided, and returns the created
        expense record as a dictionary.
        """
        # Legacy compatibility: create_expense(acct_id, paid_through_id, bill)
        if len(args) == 3 and not kwargs:
            # Legacy path: delegate to new implementation but respect provided ids
            _, account_id, paid_through_account_id = args[0], args[0], args[1]
            # args arrangement in older tests: (account_id, paid_through_account_id, bill)
            account_id = args[0]
            paid_through_account_id = args[1]
            bill = args[2]
            bill = self._validate_bill(bill)
            # legacy behavior: do not create vendor or require vendor_id
            payload: Dict[str, Any] = {
                "date": bill["date"],
                "amount": bill["amount"],
                "account_id": account_id,
                "paid_through_account_id": paid_through_account_id,
                "reference_number": bill.get("bill_no"),
                "notes": bill.get("notes") or f"Vendor: {bill.get('vendor')}",
            }
            if bill.get("currency"):
                payload["currency_code"] = bill["currency"]
            url = f"{self.base_accounts_url}/books/v3/expenses?organization_id={self.organization_id}"
            resp = self._request_with_retries("post", url, headers=self._auth_headers(), json=payload)
            if resp.status_code not in (200, 201):
                raise ZohoAPIError(f"Failed to create expense: {resp.status_code} {resp.text}")
            return resp.json()

        # New preferred signature: create_expense(bill=...)
        if len(args) == 1 and not kwargs:
            bill = args[0]
        else:
            bill = kwargs.get("bill")
        if bill is None:
            raise ZohoValidationError("create_expense expects a bill dictionary")

        # proceed with creation
        bill = self._validate_bill(bill)

        # ensure vendor
        vendor = self.create_vendor_if_not_exists(bill["vendor"])
        vendor_id = vendor.get("contact_id") or vendor.get("vendor_id") or vendor.get("contact_number")
        if not vendor_id:
            raise ZohoAPIError("Vendor record missing identifier after creation")

        # determine accounts
        account_id = bill.get("account_id")
        paid_through_account_id = bill.get("paid_through_account_id")
        if not account_id or not paid_through_account_id:
            account_id, paid_through_account_id = self._choose_account_ids(bill)

        payload: Dict[str, Any] = {
            "date": bill["date"],
            "amount": bill["amount"],
            "account_id": account_id,
            "paid_through_account_id": paid_through_account_id,
            "vendor_id": vendor_id,
            "reference_number": bill.get("bill_no"),
            "notes": bill.get("notes") or f"Vendor: {bill.get('vendor')}",
        }

        if bill.get("currency"):
            payload["currency_code"] = bill["currency"]

        if bill.get("gst"):
            payload["gst_details"] = bill["gst"]

        url = f"{self.base_accounts_url}/books/v3/expenses?organization_id={self.organization_id}"
        resp = self._request_with_retries("post", url, headers=self._auth_headers(), json=payload)
        if resp.status_code not in (200, 201):
            raise ZohoAPIError(f"Failed to create expense: {resp.status_code} {resp.text}")

        data = resp.json()
        expense = data.get("expense") or (data.get("expenses") and data.get("expenses")[0])
        if not expense:
            raise ZohoAPIError("Unexpected create expense response")

        LOGGER.info("Created expense id=%s", expense.get("expense_id"))
        return expense
