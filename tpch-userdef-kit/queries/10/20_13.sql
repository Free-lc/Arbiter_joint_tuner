-- Function Query Definition
-- Approved February 1998


select
s_name,
s_address
from
supplier_1_prt_p13,
nation_1_prt_p13
where
s_suppkey in (
select
ps_suppkey
from
partsupp_1_prt_p13
where
ps_partkey in (
select
p_partkey
from
part_1_prt_p13
where
p_name like 'forest%'
)
and ps_availqty > (
select
0.5 * sum(l_quantity)
from
lineitem_1_prt_p13
where
l_partkey = ps_partkey
and l_suppkey = ps_suppkey
and l_shipdate >= date '1994-01-01'
and l_shipdate < date '1994-01-01' + interval '1' year
)
)
and s_nationkey = n_nationkey
and n_name = 'CANADA'
order by
s_name;
limit -1;
-- $ID$
-- TPC-H/TPC-R Suppliers Who Kept Orders Waiting 
