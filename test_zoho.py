from zoho.books import ZohoBooksClient

client = ZohoBooksClient.from_env()

bill = {
    "vendor": "ABC Medicals",
    "bill_no": "INV001",
    "date": "2026-08-02",
    "amount": "250",
    "currency": "INR",
    "gst": "5%",
}

result = client.create_expense(bill)

print(result)