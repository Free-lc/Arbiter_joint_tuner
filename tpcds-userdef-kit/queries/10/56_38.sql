SELECT ws_bill_addr_sk,ws_ship_date_sk,ws_net_profit,ws_ext_discount_amt,ws_net_paid_inc_ship_tax,ws_warehouse_sk,ws_ship_customer_sk,ws_ship_addr_sk,ws_bill_customer_sk,ws_ship_mode_sk,ws_ext_list_price,ws_net_paid_inc_ship_tax,ws_ship_hdemo_sk,ws_ext_sales_price FROM web_sales_1_prt_p38 WHERE ws_item_sk Between 76297 and 81382 AND ws_ship_customer_sk Between 7 and 500000;