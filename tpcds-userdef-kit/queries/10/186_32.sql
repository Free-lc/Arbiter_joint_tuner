SELECT d_date_sk,d_date,d_fy_week_seq,d_dom,d_date_sk,d_fy_year,d_same_day_lq,d_same_day_ly,d_month_seq,d_first_dom,d_date,d_moy,d_fy_quarter_seq,d_fy_quarter_seq FROM date_dim_1_prt_p32 WHERE d_week_seq Between 6721 and 7052 AND d_qoy Between 1 and 4;