SELECT wr_account_credit,wr_returning_cdemo_sk,wr_return_quantity,wr_returned_time_sk,wr_returned_date_sk,wr_item_sk,wr_return_tax FROM web_returns_1_prt_p0 WHERE wr_reversed_charge Between 7.94 and 10.78 AND wr_refunded_hdemo_sk Between 1 and 7200;