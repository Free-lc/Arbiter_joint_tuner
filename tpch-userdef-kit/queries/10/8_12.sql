-- Functional Query Definition
-- Approved February 1998


select
o_year,
sum(case
when nation = 'BRAZIL' then volume
else 0
end) / sum(volume) as mkt_share
from
(
select
extract(year from o_orderdate) as o_year,
l_extendedprice * (1 - l_discount) as volume,
n2.n_name as nation
from
part_1_prt_p12,
supplier_1_prt_p12,
lineitem_1_prt_p12,
orders_1_prt_p12,
customer_1_prt_p12,
nation_1_prt_p12 n1,
nation_1_prt_p12 n2,
region_1_prt_p12
where
p_partkey = l_partkey
and s_suppkey = l_suppkey
and l_orderkey = o_orderkey
and o_custkey = c_custkey
and c_nationkey = n1.n_nationkey
and n1.n_regionkey = r_regionkey
and r_name = 'AMERICA'
and s_nationkey = n2.n_nationkey
and o_orderdate between date '1995-01-01' and date '1996-12-31'
and p_type = 'ECONOMY ANODIZED STEEL'
) as all_nations
group by
o_year
order by
o_year;
limit -1;
-- $ID$
-- TPC-H/TPC-R Product Type Profit Measure 
