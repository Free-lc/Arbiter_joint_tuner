SELECT inv_date_sk,inv_warehouse_sk,inv_date_sk FROM inventory WHERE inv_warehouse_sk Between 1 and 2 AND inv_quantity_on_hand Between 0 and 1000;