-- Functional Query Definition
-- Approved February 1998


select
ps_partkey,
sum(ps_supplycost * ps_availqty) as value
from
partsupp_1_prt_p41,
supplier_1_prt_p41,
nation_1_prt_p41
where
ps_suppkey = s_suppkey
and s_nationkey = n_nationkey
and n_name = 'GERMANY'
group by
ps_partkey having
sum(ps_supplycost * ps_availqty) > (
select
sum(ps_supplycost * ps_availqty) * 0.0000100000
from
partsupp_1_prt_p41,
supplier_1_prt_p41,
nation_1_prt_p41
where
ps_suppkey = s_suppkey
and s_nationkey = n_nationkey
and n_name = 'GERMANY'
)
order by
value desc;
limit -1;
-- $ID$
-- TPC-H/TPC-R Shipping Modes and Order Priority 
