SELECT wr_return_amt_inc_tax,wr_returned_time_sk,wr_returning_hdemo_sk,wr_returning_customer_sk,wr_returned_time_sk,wr_item_sk,wr_account_credit,wr_refunded_addr_sk,wr_refunded_cdemo_sk,wr_return_tax,wr_returning_cdemo_sk,wr_fee,wr_reason_sk,wr_refunded_hdemo_sk FROM web_returns_1_prt_p11 WHERE wr_returning_customer_sk Between 50747 and 51428 AND wr_refunded_customer_sk Between 60 and 99841;