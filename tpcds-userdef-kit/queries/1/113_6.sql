SELECT wr_returned_date_sk,wr_refunded_addr_sk,wr_return_ship_cost,wr_net_loss,wr_returned_date_sk,wr_returning_customer_sk,wr_fee,wr_reversed_charge,wr_returning_hdemo_sk,wr_refunded_addr_sk FROM web_returns_1_prt_p6 WHERE wr_reversed_charge Between 9.23 and 10.13 AND wr_returned_time_sk Between 29 and 85977;