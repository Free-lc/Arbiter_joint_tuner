-- Functional Query Definition
-- Approved February 1998


select
sum(l_extendedprice) / 7.0 as avg_yearly
from
lineitem_1_prt_p48,
part_1_prt_p48
where
p_partkey = l_partkey
and p_brand = 'Brand#23'
and p_container = 'MED BOX'
and l_quantity < (
select
0.2 * avg(l_quantity)
from
lineitem_1_prt_p48
where
l_partkey = p_partkey
);
limit -1;
-- $ID$
-- TPC-H/TPC-R Large Volume Customer 
