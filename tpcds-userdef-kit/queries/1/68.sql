SELECT cr_return_amount,cr_reason_sk,cr_refunded_customer_sk,cr_refunded_hdemo_sk,cr_returning_addr_sk,cr_refunded_cdemo_sk,cr_refunded_cash,cr_returning_addr_sk,cr_reversed_charge,cr_warehouse_sk,cr_return_amt_inc_tax,cr_return_quantity,cr_returning_hdemo_sk,cr_store_credit FROM catalog_returns WHERE cr_refunded_cdemo_sk Between 1299380 and 1348346 AND cr_returning_cdemo_sk Between 1733 and 1920740;