-- Functional Query Definition
-- Approved February 1998

create view revenue0 (supplier_no, total_revenue) as
select
l_suppkey,
sum(l_extendedprice * (1 - l_discount))
from
lineitem_1_prt_p32
where
l_shipdate >= date '1996-01-01'
and l_shipdate < date '1996-01-01' + interval '3' month
group by
l_suppkey;


select
s_suppkey,
s_name,
s_address,
s_phone,
total_revenue
from
supplier_1_prt_p32,
revenue0
where
s_suppkey = supplier_no
and total_revenue = (
select
max(total_revenue)
from
revenue0
)
order by
s_suppkey;

drop view revenue0;
limit -1;
-- $ID$
-- TPC-H/TPC-R Parts/Supplier Relationship 
