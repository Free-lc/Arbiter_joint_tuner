SELECT d_date,d_quarter_seq,d_dom,d_date_sk,d_moy,d_year,d_last_dom,d_first_dom,d_week_seq,d_fy_quarter_seq,d_date_sk,d_dow,d_week_seq,d_fy_week_seq,d_first_dom,d_same_day_ly FROM date_dim_1_prt_p21 WHERE d_fy_year Between 1975 and 1983 AND d_date_sk Between 2442728 and 2445468;