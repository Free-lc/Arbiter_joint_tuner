SELECT wr_return_tax,wr_returning_customer_sk,wr_returned_date_sk,wr_return_amt,wr_return_tax,wr_return_amt,wr_returning_addr_sk,wr_returning_addr_sk,wr_return_amt_inc_tax,wr_account_credit,wr_returning_hdemo_sk,wr_refunded_cash,wr_return_amt,wr_net_loss,wr_refunded_hdemo_sk,wr_item_sk FROM web_returns_1_prt_p44 WHERE wr_refunded_addr_sk Between 86602 and 98312 AND wr_net_loss Between 1.34 and 12250.22;