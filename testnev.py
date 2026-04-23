import os
import requests

GROQ_API_KEY = os.getenv("Groq_key")

def groq_chat(user_query):
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
    "role": "system",
    "content": """
You are a strict document-based question answering assistant.

You will be given a CONTEXT (policy document). You must answer ONLY using this context.make sure if user query matches any of context details give all details dont leave anything in context

Rules:
1. If the answer is NOT in the context, say: "Answer not found in provided document."
2. Do NOT use outside knowledge.
3. Do NOT guess or assume anything.
4. Be short, clear, and accurate.
5. If user question matches a section in context, extract and summarize only that part.
6. Do not repeat full document.

CONTEXT:
(Returns & refunds for seller fulfilled orders from Amazon.in)
PDF_CONTEXT =
==============================
RETURNS & REFUNDS POLICY
SELLER FULFILLED ORDERS (AMAZON.IN)
==============================

Returns & refunds for seller fulfilled orders
When you buy an item from a seller on Amazon.in, you should be able to expect a high standard 
of customer service, both before and after you receive your order.

==============================
FREQUENTLY ASKED QUESTIONS
==============================

 What's the returns policy for seller fulfilled items?
 How do I return seller fulfilled items?
 Who bears the delivery charges for returning seller fulfilled items?
 How do I get a refund?
 How are refunds for seller fulfilled orders calculated?
 What do I do if the seller doesn't respond to my return request?

==============================
RETURNS POLICY
==============================

What's the returns policy for seller fulfilled items?
The following returns policy applies to seller fulfilled items purchased on Amazon.

Product Categories:

Return Time from Delivery:
- Books → 30 Days (damaged/defective), 7 Days (no longer want/need)
- Movies & TV → 30 Days (damaged/defective), 7 Days (no longer want/need)
- Electronics → 10 Days (damaged/defective), 10 Days (no longer want/need)
- Kindle → Visit Kindle help pages for Kindle-specific returns policy

==============================
RETURN CONDITION
==============================

Damaged/ Defective Items

For Books, Movies & TV:
 All damaged/ defective items must be returned in the original condition they were received in.
 For returns of damaged/ defective items, you need to contact the seller within 14 days of delivery to inform them of the damage/ defect.

For Electronics:
 You can return electronics that are dead on arrival, or arrive in damaged or defective condition.
 Such items must be returned in the same condition you received them in with all the accessories.
 These items must also have the serial number and UPC intact.
 The seller will either get the defective product repaired at the service centre or send a replacement.
 The seller will not charge additional shipping charges for sending the repaired product or a replacement item.
 In case of returns of "dead on arrival" and damaged items, the seller reserves the right to test and inspect the product for customer misuse and issue a replacement after inspection.
 If the item cannot be repaired or replaced, the seller will issue a refund.

==============================
ITEMS YOU NO LONGER WANT / NEED
==============================

 Books you no longer want/ need must be returned in new condition.
 Items from Movies & TV category must be returned in new and unopened condition with all accessories and original packing unopened and intact.
 All Electronics which includes the categories: Mobiles & Accessories, Cameras & Photography, Portable Media Players, Car & Vehicle Electronics, Computers, Components & Accessories, Hi-fi & Home Audio, Home Theatre, TV & Video and Telephones must be returned in new and unopened condition with all accessories and original packing intact and unopened.
 These items must also have the serial number or UPC intact.
 The UPC number is a sequence of 12 digits below the barcode that is uniquely assigned to a product.
 All accessories from Electronics category must be returned in new and unopened condition with original packing intact.

Return to Frequently Asked Questions or go back to Help

==============================
HOW DO I RETURN SELLER FULFILLED ITEMS?
==============================

Seller fulfilled items need to be returned directly to the seller and not to Amazon. Follow these steps:

1. Click Your Orders
2. Find the order you want to return and click the Contact Seller button next to the order
3. Choose the appropriate reason from the drop-down menu and click Continue
4. Enter your query and click Send e-mail. Your query will be forwarded to the seller

The seller will send you an e-mail with their return address and mailing instructions if the return is approved.

You can see sellers' responses to your requests in the Returns Support Centre.

Please allow 3 business days to give your seller a chance to respond to your return request and provide instructions on returning your item.

We recommend using a traceable delivery service to return the item.

Return to Frequently Asked Questions or go back to Help

==============================
DELIVERY CHARGES
==============================

Who bears the delivery charges for returning seller fulfilled items?

Delivery and handling charges for returning unwanted items dispatched by a seller are usually borne by the buyer.

If you need to return an item due to a fault on the part of the seller, you can request that they also reimburse you for return postage costs.

To ensure safe return of the merchandise, it is recommended that items should be returned to the seller using a traceable delivery service.

Return to Frequently Asked Questions or go back to Help

==============================
REFUNDS
==============================

How do I get a refund?

To get refunds for seller fulfilled orders, initiate the returns process by contacting the seller from the Your Orders page.

Remember that the items have to be returned to the sellers, not to Amazon.in.

In any case where a refund is required, the sellers need to authorise that refund.

Amazon.in can assist in facilitating refunds for you only when the seller notifies us of the receipt of the item.

For Electronics, if the product was delivered to you in a damaged or defective condition, or was dead on arrival, the seller will issue a refund if the product cannot be repaired or replaced.

Return to Frequently Asked Questions or go back to Help

==============================
REFUND CALCULATION
==============================

How are refunds for seller fulfilled orders calculated?

When you're returning an item that was damaged or defective, the seller will refund the cost of the item as well as the delivery charges incurred in sending it to you.

For returns of items that you no longer want/ need, sellers are not required to refund the return delivery charges incurred, and other services provided to you in connection with the purchase, unless the return is due to the sellers' error.

Return to Frequently Asked Questions or go back to Help

==============================
NO RESPONSE FROM SELLER
==============================

What do I do if the seller doesn't respond to my return request?

Please allow 3 business days for your seller to address the issue and provide instructions for your return.

If the seller does not respond even after 3 business days, or if your issue is not addressed to your satisfaction, you can submit an A-to-z Guarantee claim to Amazon.

Amazon will stand by your claim and help you recover the payments made to the seller if the claim is valid.

Return to Frequently Asked Questions or go back to Help

Your job: Answer user questions ONLY using this context.
"""
},
            {
                "role": "user",
                "content": user_query
            }
        ],
        "temperature": 0.2
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.text}"


# ---------------- TEST ----------------
if __name__ == "__main__":
    while True:
        q = input("Ask: ")
        print("\nAnswer:", groq_chat(q))
        print("-" * 50)