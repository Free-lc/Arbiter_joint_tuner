SELECT ws_bill_cdemo_sk,ws_wholesale_cost,ws_ext_discount_amt,ws_coupon_amt,ws_sold_time_sk,ws_promo_sk,ws_quantity,ws_web_site_sk FROM web_sales_1_prt_p14 WHERE ws_ext_list_price Between 9186.66 and 9672.62 AND ws_ship_mode_sk Between 1 and 20;