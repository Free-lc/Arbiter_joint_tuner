-- Functional Query Definition
-- Approved February 1998


select
100.00 * sum(case
when p_type like 'PROMO%'
then l_extendedprice * (1 - l_discount)
else 0
end) / sum(l_extendedprice * (1 - l_discount)) as promo_revenue
from
lineitem_1_prt_p6,
part_1_prt_p6
where
l_partkey = p_partkey
and l_shipdate >= date '1995-09-01'
and l_shipdate < date '1995-09-01' + interval '1' month;
limit -1;
-- $ID$
-- TPC-H/TPC-R Top Supplier 
