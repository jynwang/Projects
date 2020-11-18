---------------------------------------------------------------------
----------------------seperate the raw data--------------------------
---------------------------------------------------------------------
-- trade
if (object_id('data') is not null) drop table data
create table data
(
Raw_data varchar(150),
Time_ bigint,
Exchange varchar(1),
Symbol varchar(16),
Sale_Condition varchar(4),
Trade_Volume varchar(9),
Trade_Price float,
Trade_Stop_Stock_Indicator varchar(1),
Trade_Correlation_Indicator varchar(2),
Trade_Sequency_Number varchar(16)
)
go 

insert into data
select *, substring(rawdata,1, 9),
SUBSTRING(rawdata, 10, 1), 
SUBSTRING(rawdata, 11, 16), 
SUBSTRING(rawdata, 27, 4),
SUBSTRING(rawdata, 31, 9),
cast(SUBSTRING(rawdata, 40, 11) as float)/10000,
SUBSTRING(rawdata, 51, 1),
SUBSTRING(rawdata, 52, 2),
SUBSTRING(rawdata, 54, 16)
from trade_initial;
go

alter table data 
drop column Raw_data
go

--nbbo
if (object_id('data_n') is not null) drop table data_n
create table data_n
(
Raw_data varchar(300),
Time_ bigint,
Exchange varchar(1),
Symbol varchar(16),
Bid_Price float,
Bid_Size bigint,
Ask_Price float,
Ask_Size bigint,
Quote_Condition varchar(1),
Bid_Exchange varchar(1),
Ask_Exchange varchar(1),
Sequence_Number varchar(16),
Best_Bid_Exchange varchar(1),
Best_Bid_Price float,
Best_Bid_Size bigint,
Best_Offer_Exchange varchar(1),
Best_Offer_Price float,
Best_Offer_Size bigint
)
go

insert into data_n
select *, SUBSTRING(rawdata, 1, 9), --time
SUBSTRING(rawdata, 10, 1), 
SUBSTRING(rawdata, 11, 16), 
cast(SUBSTRING(rawdata,27,11) as float)/10000,--price
SUBSTRING(rawdata,38,7),
cast(SUBSTRING(rawdata,45,11) as float)/10000,--price
SUBSTRING(rawdata,56,7),
SUBSTRING(rawdata,63,1),
SUBSTRING(rawdata,68,1),
SUBSTRING(rawdata,69,1),
SUBSTRING(rawdata,70,16),
SUBSTRING(rawdata,91,1),
cast(SUBSTRING(rawdata,92,11) as float)/10000,
SUBSTRING(rawdata,103,7),
SUBSTRING(rawdata,117,1),
cast(SUBSTRING(rawdata,118,11) as float)/10000,
SUBSTRING(rawdata,129,7)
from nbbo_initial;
go

alter table data_n drop column Raw_data
go

/* might need later
drop table nbbo_initial
drop table trade_initial
*/

---------------------------------------------------------------------
----------------------common selection table-------------------------
---------------------------------------------------------------------
/*Maybel we could run this in a different script, 
since we only need to run this once then it stays the same*/

--ID List
if (object_id('ID_list') is not null) drop table ID_list
create table ID_list
(
Id int,
Symbol varchar(5)
)
go

insert into ID_list
values	(1, 'AAPL'), (2, 'XOM'), (3, 'GOOG'), (4, 'MSFT'), (5, 'JNJ'), (6, 'WFC'), (7, 'GE'), (8, 'WMT'), (9, 'CVX'), 
		(10, 'JPM')
go

--The table time_intervals include all the possible intervals in the day */

if (object_id('time_intervals') is not null) drop table time_intervals
create table time_intervals 
(
begin_interval bigint, 
end_interval bigint
)
go

insert into time_intervals
values /*(90000000, 90500000), (90500000, 91000000), (91000000, 91500000), (91500000, 92000000), (92000000, 92500000), (92500000, 93000000)*/
	   (93000000, 93500000), (93500000, 94000000), (94000000, 94500000), (94500000, 95000000), (95000000, 95500000), (95500000, 100000000),
	   (100000000, 100500000), (100500000, 101000000), (101000000, 101500000), (101500000, 102000000), (102000000, 102500000), (102500000, 103000000),
	   (103000000, 103500000), (103500000, 104000000), (104000000, 104500000), (104500000, 105000000), (105000000, 105500000), (105500000, 110000000),
	   (110000000, 110500000), (110500000, 111000000), (111000000, 111500000), (111500000, 112000000), (112000000, 112500000), (112500000, 113000000),
	   (113000000, 113500000), (113500000, 114000000), (114000000, 114500000), (114500000, 115000000), (115000000, 115500000), (115500000, 120000000),
	   (120000000, 120500000), (120500000, 121000000), (121000000, 121500000), (121500000, 122000000), (122000000, 122500000), (122500000, 123000000),
	   (123000000, 123500000), (123500000, 124000000), (124000000, 124500000), (124500000, 125000000), (125000000, 125500000), (125500000, 130000000),
	   (130000000, 130500000), (130500000, 131000000), (131000000, 131500000), (131500000, 132000000), (132000000, 132500000), (132500000, 133000000),
	   (133000000, 133500000), (133500000, 134000000), (134000000, 134500000), (134500000, 135000000), (135000000, 135500000), (135500000, 140000000),
	   (140000000, 140500000), (140500000, 141000000), (141000000, 141500000), (141500000, 142000000), (142000000, 142500000), (142500000, 143000000),
	   (143000000, 143500000), (143500000, 144000000), (144000000, 144500000), (144500000, 145000000), (145000000, 145500000), (145500000, 150000000),
	   (150000000, 150500000), (150500000, 151000000), (151000000, 151500000), (151500000, 152000000), (152000000, 152500000), (152500000, 153000000),
	   (153000000, 153500000), (153500000, 154000000), (154000000, 154500000), (154500000, 155000000), (155000000, 155500000), (155500000, 160000000)
go

--Exchange List
--if (object_id('exchange_list') is not null) drop table exchange_list
--create table exchange_list
--(
--Id int,
--Exchange varchar(1)
--)
--go

--insert into exchange_list
--values (1, 'A'), (2, 'B'), (3, 'C'), (4, 'D'), (5, 'J'), (6, 'K'), (7, 'M'), (8, 'N'), (9, 'P'), (10, 'Q'), (11, 'T'), (12, 'W'), (13, 'X'), (14, 'Y'), (15, 'Z')
--go

---------------------------------------------------------------------
--------------------------Select 51 stocks---------------------------
---------------------------------------------------------------------
-- trade
if (object_id('trade') is not null) drop table trade
create table trade
(
Time_ bigint,
Exchange varchar(1),
Symbol varchar(16),
Sale_Condition varchar(4),
Trade_Volume bigint,
Trade_Price float,
Trade_Stop_Stock_Indicator varchar(1),
Trade_Correlation_Indicator varchar(2),
Trade_Sequency_Number varchar(16)
)
go

insert into trade
select	
	case when a.Sale_Condition like '%O%' and a.Time_ < 93000000 then 93000000 else a.Time_ end, --time
	a.Exchange, 
	a.Symbol, a.Sale_Condition, a.Trade_Volume, a.Trade_Price, 
	a.Trade_Stop_Stock_Indicator, a.Trade_Correlation_Indicator, a.Trade_Sequency_Number
from data a
inner join ID_list b
on a.Symbol = b.Symbol
where 
a.Time_ >= 90000000 and a.Time_ <= 160000000
and (a.Sale_Condition like '[^L]'
and a.Sale_Condition like '[^U]'
and a.Sale_Condition like '[^Z]'
and a.Sale_Condition like '[^K]'
and a.Sale_Condition like '[^4]'
and a.Sale_Condition like '[^P]'
and a.Sale_Condition like '[^W]')
or  (a.Sale_Condition like '%F%' or a.Sale_Condition like '%O%' or a.Sale_Condition like '%Q%');
go

--select top 100 * from trade

--nbbo 
--2 step, 1: select, 2: add in the next point of time as a preperation for TWAS
if (object_id('nbbo') is not null) drop table nbbo
create table nbbo
(
ID_num int IDENTITY(1,1),
Time_ bigint,
Exchange varchar(1),
Symbol varchar(16),
Bid_Price float,
Bid_Size bigint,
Ask_Price float,
Ask_Size bigint,
Quote_Condition varchar(1),
Bid_Exchange varchar(1),
Ask_Exchange varchar(1),
Sequence_Number varchar(16),
Best_Bid_Exchange varchar(1),
Best_Bid_Price float,
Best_Bid_Size bigint,
Best_Offer_Exchange varchar(1),
Best_Offer_Price float,
Best_Offer_Size bigint,
Spread float,
Quote_Mid float
)
go

insert into nbbo
select
	a.Time_, --time
	a.Exchange,
	a.Symbol,
	a.Bid_Price,
	a.Bid_Size,
	a.Ask_Price,
	a.Ask_Size,
	a.Quote_Condition,
	a.Bid_Exchange,
	a.Ask_Exchange,
	a.Sequence_Number,
	a.Best_Bid_Exchange,
	a.Best_Bid_Price,
	a.Best_Bid_Size,
	a.Best_Offer_Exchange,
	a.Best_Offer_Price,
	a.Best_Offer_Size,
	(a.Best_Offer_Price-a.Best_Bid_Price) as Spread,
	(a.Best_Bid_Price+a.Best_Offer_Price)/2 as Quote_Mid
from data_n a
inner join ID_list b
on a.Symbol = b.Symbol
where 
a.Time_ >= 90000000 and a.Time_ <= 160000000
order by a.Symbol,/*a.Exchange,*/a.Time_
go

--prepare for TWAS
if (object_id('nbbo_b') is not null) drop table nbbo_b
create table nbbo_b
(
Time_1 bigint,
Time_2 bigint,
Exchange varchar(1),
Symbol varchar(16),
Bid_Price float,
Bid_Size bigint,
Ask_Price float,
Ask_Size bigint,
Quote_Condition varchar(1),
Bid_Exchange varchar(1),
Ask_Exchange varchar(1),
Sequence_Number varchar(16),
Best_Bid_Exchange varchar(1),
Best_Bid_Price float,
Best_Bid_Size bigint,
Best_Offer_Exchange varchar(1),
Best_Offer_Price float,
Best_Offer_Size bigint,
Spread float,
Quote_Mid float
)
go

insert into nbbo_b
select 
	a.Time_, --time 1
	b.Time_, --time 2
	a.Exchange,
	a.Symbol,
	a.Bid_Price,
	a.Bid_Size,
	a.Ask_Price,
	a.Ask_Size,
	a.Quote_Condition,
	a.Bid_Exchange,
	a.Ask_Exchange,
	a.Sequence_Number,
	a.Best_Bid_Exchange,
	a.Best_Bid_Price,
	a.Best_Bid_Size,
	a.Best_Offer_Exchange,
	a.Best_Offer_Price,
	a.Best_Offer_Size,
	(a.Best_Offer_Price-a.Best_Bid_Price) as Spread,
	(a.Best_Bid_Price+a.Best_Offer_Price)/2 as Quote_Mid
from nbbo a
inner join nbbo b
on a.Symbol = b.Symbol
--and a.Exchange = b.Exchange
and a.ID_num + 1 = b.ID_num
go

--select * from nbbo_b

/* might need later
drop table data
drop table data_n
drop table nbbo
*/

---------------------------------------------------------------------
--------------------------Calculate statss---------------------------
------------------------Using trade and nbbo_b-----------------------
---------------------------------------------------------------------

-- trade
-- 2 steps: 1: part of stats, 2: add in return
if (object_id('trade_stats_a') is not null) drop table trade_stats_a
create table trade_stats_a
(
begin_interval bigint,
end_interval bigint,
Symbol varchar(16),
--Exchange varchar(1),
Average_Price float,
--Last_Time bigint,
Last_Price float,
High_Price float,
Low_Price float,
Price_Range float,
Price_Movement float,
Trade_Volume bigint,
Dollar_Volume float,
Trade_Count bigint,
Volume_Weighted_Price float
)
go

insert into trade_stats_a
select			b.begin_interval,
                b.end_interval,
                a.Symbol, 
				--a.Exchange,
                AVG(a.Trade_Price) as Average_Price,
				--Max(a.Time_) as Last_Time,  -- in case we have to calculate the aggregated last price for one symbol
                d.Trade_Price as Last_Price, --/* last traded price in the interval */
                MAX(a.Trade_Price) as High_Price, 
                MIN(a.Trade_Price) as Low_Price,
                MAX(a.Trade_Price)-MIN(a.Trade_Price) as Price_Range,
                2*(MAX(a.Trade_Price) - MIN(a.Trade_Price))/(MAX(a.Trade_Price) + MIN(a.Trade_Price)) as Price_Movement,   --don't understand why it's the movement: (U-L)/middle             
                SUM(a.Trade_Volume) as Trade_Volume,
                SUM(a.Trade_Volume*a.Trade_Price) as Dollar_Volume,
                COUNT(a.Symbol) as Trade_Count,
                SUM(a.Trade_Volume*a.Trade_Price)/SUM(a.Trade_Volume) as Volume_Weighted_Price
from trade a
inner join time_intervals b
on a.Time_ >= b.begin_interval /* greater than or equal to beginning of 5-min interval */
and a.Time_ < b.end_interval /* less than end of 5-min interval*/
--inner join exchange_list c
--on a.Exchange = c.Exchange
inner join trade d
on a.Symbol = d.Symbol
--and a.Exchange = d.Exchange
group by b.begin_interval,b.end_interval,a.Symbol, d.time_,d.Trade_Price,d.Trade_Sequency_Number--,a.Exchange
having d.Trade_Sequency_Number = Max(a.Trade_Sequency_Number) --changed!!!!!

-- add in returns
if (object_id('trade_stats') is not null) drop table trade_stats
create table trade_stats
(
begin_interval bigint,
end_interval bigint,
Symbol varchar(16),
--Exchange varchar(1),
Average_Price float,
High_Price float,
Low_Price float,
--Last_Time bigint,
Price_Range float,
Price_Movement float,
Trade_Volume bigint,
Dollar_Volume float,
Trade_Count bigint,
Volume_Weighted_Price float,
Return_ float
)
go

insert into trade_stats
select			a.begin_interval, a.end_interval, a.Symbol, --a.Exchange,
                a.Average_Price, a.High_Price, a.Low_Price,/*a.Last_Time,*/ a.Price_Range, a.Price_Movement,            
                a.Trade_Volume, a.Dollar_Volume, a.Trade_Count, a.Volume_Weighted_Price,
                LOG(a.Last_Price/b.Last_Price) as Return_ /* Calculate return */
from trade_stats_a a /* current time interval */
left join trade_stats_a b /* previous time interval */
on a.Symbol = b.Symbol
--and a.Exchange = b.Exchange
and (a.begin_interval - b.begin_interval = 500000 or (a.begin_interval - b.begin_interval = 4500000 and a.begin_interval%10000000=0)) 
group by        a.begin_interval, a.end_interval, a.Symbol, --a.Exchange,
                a.Average_Price, a.High_Price, /*a.Last_Time,*/ a.Low_Price, a.Price_Range, a.Price_Movement,            
                a.Trade_Volume, a.Dollar_Volume, a.Trade_Count, a.Volume_Weighted_Price,a.Last_Price,b.Last_Price
--nbbo
if (object_id('nbbo_stats') is not null) drop table nbbo_stats
create table nbbo_stats
(
Begin_Interval bigint,
End_Interval bigint,
Symbol varchar(16),
--Exchange varchar(1),
Time_Weigthed_Spread float,
Volume_Weighted_Spread float,
Effective_Spread float
)
go

insert into nbbo_stats
select			d.begin_interval,
                d.end_interval,
                a.Symbol, 
				--a.Exchange,
				sum((a.Time_2/10000000 *3600000 + a.Time_2 %10000000/100000 * 60000 + a.Time_2 %100000
				-(a.Time_1/10000000 *3600000 + a.Time_1%10000000/100000 * 60000 + a.Time_1%100000))*a.Spread)/300000 as Time_Weigthed_Spread,
				sum(b.Trade_Volume*a.Spread)/sum(b.Trade_Volume) as Volume_Weighted_Spread,
				avg(abs(b.Trade_Price-a.Quote_Mid)*2) as Effective_Spread
from nbbo_b a
inner join trade b
on a.Time_1 = b.Time_ 
and a.Symbol = b.Symbol
--inner join exchange_list c
--on a.Exchange = c.Exchange
inner join time_intervals d
on a.Time_1 >= d.begin_interval 
and a.Time_1 < d.end_interval 
group by d.begin_interval,d.end_interval,a.Symbol--,a.Exchange


---------------------------------------------------------------------
-------------------Combine all Stats in one file---------------------
---------------------------------------------------------------------

if (object_id('stats') is not null) drop table stats
create table stats
(
begin_interval bigint,
end_interval bigint,
Symbol varchar(16),
--Exchange varchar(1),
Average_Price float,
High_Price float,
Low_Price float,
Price_Range float,
Price_Movement float,
Trade_Volume bigint,
Dollar_Volume float,
Trade_Count bigint,
Volume_Weighted_Price float,
Return_ float,
Time_Weigthed_Spread float,
Volume_Weighted_Spread float,
Effective_Spread float
)
go

insert into stats
select a.*,b.Time_Weigthed_Spread,b.Volume_Weighted_Spread,b.Effective_Spread
from trade_stats a
full join nbbo_stats b
on a.Symbol = b.Symbol
--and a.Exchange = b.Exchange
and a.begin_interval = b.Begin_Interval
go

--select top 100 * from stats