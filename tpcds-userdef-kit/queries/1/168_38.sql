SELECT sr_return_tax,sr_customer_sk,sr_return_amt,sr_return_amt_inc_tax,sr_store_sk,sr_returned_date_sk,sr_return_quantity FROM store_returns_1_prt_p38 WHERE sr_ticket_number Between 223424 and 227389 AND sr_return_amt Between 0.0 and 13220.48;