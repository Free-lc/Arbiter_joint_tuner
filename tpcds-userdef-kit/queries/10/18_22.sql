SELECT ss_ticket_number,ss_customer_sk,ss_list_price,ss_store_sk,ss_addr_sk,ss_net_paid_inc_tax,ss_addr_sk,ss_ext_discount_amt,ss_item_sk,ss_promo_sk,ss_sold_date_sk,ss_net_paid,ss_wholesale_cost,ss_hdemo_sk,ss_net_paid_inc_tax,ss_sold_time_sk,ss_net_profit,ss_ext_wholesale_cost,ss_ext_sales_price,ss_net_paid_inc_tax FROM store_sales_1_prt_p22 WHERE ss_sold_date_sk Between 2451623 and 2451728 AND ss_ext_list_price Between 1.21 and 19798.0;