-- Functional Query Definition
-- Approved February 1998


select
c_count,
count(*) as custdist
from
(
select
c_custkey,
count(o_orderkey) c_count
from
customer_1_prt_p15 left outer join orders_1_prt_p15 on
c_custkey = o_custkey
and o_comment not like '%special%requests%'
group by
c_custkey
)
group by
c_count
order by
custdist desc,
c_count desc;
limit -1;
-- $ID$
-- TPC-H/TPC-R Promotion Effect 
