SELECT cr_refunded_cash,cr_ship_mode_sk,cr_store_credit,cr_item_sk,cr_returning_cdemo_sk,cr_return_amount,cr_return_amt_inc_tax,cr_refunded_cdemo_sk,cr_item_sk,cr_returned_time_sk,cr_warehouse_sk,cr_return_amt_inc_tax,cr_returned_date_sk,cr_refunded_hdemo_sk,cr_return_quantity,cr_call_center_sk,cr_refunded_customer_sk,cr_return_quantity,cr_reason_sk FROM catalog_returns_1_prt_p9 WHERE cr_return_quantity Between 15 and 17 AND cr_ship_mode_sk Between 1 and 20;