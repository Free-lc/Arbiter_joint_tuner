SELECT ws_bill_customer_sk,ws_ext_list_price,ws_quantity,ws_promo_sk,ws_bill_addr_sk,ws_order_number,ws_sales_price,ws_bill_cdemo_sk,ws_ext_discount_amt,ws_ship_mode_sk,ws_item_sk,ws_ship_hdemo_sk,ws_net_profit,ws_ext_sales_price,ws_net_paid,ws_bill_customer_sk,ws_ship_customer_sk,ws_promo_sk,ws_web_page_sk,ws_ship_addr_sk,ws_ship_mode_sk FROM web_sales_1_prt_p19 WHERE ws_net_paid_inc_tax Between 1384.0 and 1683.86 AND ws_web_page_sk Between 1 and 60;