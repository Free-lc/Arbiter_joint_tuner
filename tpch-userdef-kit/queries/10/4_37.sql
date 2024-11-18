-- Functional Query Definition
-- Approved February 1998


select
o_orderpriority,
count(*) as order_count
from
orders_1_prt_p37
where
o_orderdate >= date '1993-07-01'
and o_orderdate < date '1993-07-01' + interval '3' month
and exists (
select
*
from
lineitem_1_prt_p37
where
l_orderkey = o_orderkey
and l_commitdate < l_receiptdate
)
group by
o_orderpriority
order by
o_orderpriority;
limit -1;
-- $ID$
-- TPC-H/TPC-R Local Supplier Volume 
