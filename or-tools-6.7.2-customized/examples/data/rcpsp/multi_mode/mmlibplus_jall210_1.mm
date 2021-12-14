jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 4 R
- nonrenewable              : 4 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	1		2 
2	6	4		10 9 7 3 
3	6	3		8 5 4 
4	6	3		14 12 6 
5	6	2		15 6 
6	6	6		23 18 17 16 13 11 
7	6	6		23 18 17 16 13 11 
8	6	7		30 28 25 24 22 18 17 
9	6	4		28 24 19 14 
10	6	7		30 28 25 24 23 22 19 
11	6	7		30 29 28 25 24 20 19 
12	6	5		30 28 23 22 17 
13	6	6		30 29 28 24 22 19 
14	6	5		30 27 25 23 22 
15	6	5		30 29 28 22 21 
16	6	4		29 25 24 21 
17	6	2		29 19 
18	6	2		29 20 
19	6	1		21 
20	6	1		21 
21	6	2		27 26 
22	6	6		38 37 34 33 32 31 
23	6	6		38 37 34 33 31 29 
24	6	2		31 26 
25	6	2		31 26 
26	6	6		38 36 35 34 33 32 
27	6	4		40 38 37 31 
28	6	5		40 39 37 36 33 
29	6	5		45 40 39 36 35 
30	6	3		41 36 34 
31	6	3		39 36 35 
32	6	5		51 45 43 41 39 
33	6	5		51 45 44 43 41 
34	6	5		51 45 43 42 40 
35	6	4		51 44 43 41 
36	6	4		51 49 43 42 
37	6	3		45 44 43 
38	6	5		51 50 49 47 46 
39	6	3		50 48 44 
40	6	2		46 44 
41	6	1		42 
42	6	3		50 48 47 
43	6	2		50 46 
44	6	2		49 47 
45	6	2		49 47 
46	6	1		48 
47	6	1		52 
48	6	1		52 
49	6	1		52 
50	6	1		52 
51	6	1		52 
52	1	0		
************************************************************************
REQUESTS/DURATIONS
jobnr.	mode	dur	R1	R2	R3	R4	N1	N2	N3	N4	
------------------------------------------------------------------------
1	1	0	0	0	0	0	0	0	0	0	
2	1	3	25	12	12	25	21	20	26	24	
	2	5	24	10	12	23	19	20	26	24	
	3	6	23	9	12	18	19	14	26	22	
	4	8	23	7	11	11	17	13	26	22	
	5	9	21	5	11	10	16	8	26	21	
	6	19	20	4	11	3	16	3	26	20	
3	1	13	25	12	3	22	24	20	29	28	
	2	14	25	9	2	21	23	18	26	25	
	3	15	25	7	2	19	21	16	22	22	
	4	16	25	7	2	17	17	15	18	21	
	5	20	25	5	1	15	17	14	16	18	
	6	21	25	4	1	14	15	11	14	17	
4	1	1	12	27	20	5	23	29	8	23	
	2	3	10	22	19	5	22	29	6	22	
	3	6	8	21	18	4	21	29	5	22	
	4	17	8	19	17	4	21	29	3	22	
	5	25	6	17	15	4	19	29	3	22	
	6	26	4	15	13	3	19	29	1	22	
5	1	2	17	27	16	25	11	27	27	27	
	2	6	15	27	15	22	9	26	23	26	
	3	9	13	27	15	22	9	23	19	24	
	4	21	11	27	13	17	5	20	17	22	
	5	22	9	27	13	16	4	17	13	20	
	6	30	7	27	12	10	2	15	10	20	
6	1	13	4	9	28	25	5	16	21	26	
	2	17	4	7	28	22	4	15	19	23	
	3	20	4	7	26	19	4	14	16	22	
	4	21	3	4	24	14	4	13	15	16	
	5	24	3	4	23	11	4	11	11	12	
	6	26	2	3	22	11	4	11	11	11	
7	1	2	15	26	22	13	24	19	25	12	
	2	10	11	26	21	11	24	15	25	11	
	3	12	9	26	21	9	22	11	25	10	
	4	22	9	25	21	9	22	9	25	9	
	5	23	6	25	21	7	19	9	24	8	
	6	30	5	25	21	5	19	6	24	7	
8	1	3	25	24	11	22	25	28	27	22	
	2	6	21	19	11	20	25	27	21	21	
	3	18	17	19	11	17	25	23	18	21	
	4	19	14	16	11	17	24	21	12	20	
	5	20	12	13	11	15	23	19	7	20	
	6	27	8	13	11	14	23	19	7	20	
9	1	3	30	22	21	16	10	29	24	2	
	2	10	27	21	16	16	9	21	23	2	
	3	13	24	17	14	14	9	17	21	2	
	4	15	20	13	12	14	9	15	20	2	
	5	24	18	12	6	12	8	10	20	2	
	6	30	17	10	5	12	8	5	18	2	
10	1	4	29	18	8	25	23	19	25	26	
	2	8	29	18	8	20	22	17	21	26	
	3	11	29	17	8	17	22	14	21	24	
	4	14	29	14	8	13	20	11	18	24	
	5	27	29	14	8	8	20	9	14	23	
	6	28	29	11	8	1	19	6	10	22	
11	1	2	22	6	13	11	8	22	14	27	
	2	4	19	6	13	11	8	22	14	25	
	3	16	17	6	12	8	8	21	13	24	
	4	19	13	5	12	7	8	21	11	24	
	5	20	10	5	11	6	8	20	10	23	
	6	25	6	5	11	4	8	20	9	22	
12	1	1	16	18	26	6	12	23	27	15	
	2	22	13	14	24	6	10	21	25	15	
	3	23	11	11	20	5	8	18	21	15	
	4	24	8	9	15	3	7	14	21	15	
	5	26	4	8	13	3	5	9	19	15	
	6	28	3	6	10	1	1	5	16	15	
13	1	1	10	24	22	16	17	18	21	25	
	2	2	8	22	21	13	16	17	21	21	
	3	9	7	16	21	10	15	14	17	19	
	4	13	6	14	21	8	15	11	15	17	
	5	28	5	10	21	7	13	9	10	14	
	6	29	5	9	21	6	13	8	6	10	
14	1	6	30	19	24	22	12	10	25	15	
	2	15	26	17	19	21	12	10	20	13	
	3	16	18	16	14	21	11	10	19	13	
	4	21	16	15	12	21	9	10	16	11	
	5	23	10	14	9	20	9	9	12	10	
	6	29	8	14	2	20	7	9	6	8	
15	1	1	27	16	21	25	30	30	4	11	
	2	5	25	16	18	23	23	21	3	11	
	3	11	22	16	12	21	21	17	2	11	
	4	17	18	16	9	18	15	16	2	11	
	5	21	16	16	8	16	14	10	1	11	
	6	29	9	16	3	14	8	4	1	11	
16	1	6	26	21	11	17	20	17	10	27	
	2	8	22	19	10	16	19	16	10	25	
	3	11	20	15	10	13	18	13	10	24	
	4	21	18	12	9	9	18	13	10	23	
	5	22	17	11	9	8	16	11	10	22	
	6	30	14	7	9	5	16	9	10	22	
17	1	10	27	23	8	22	16	28	13	21	
	2	11	27	19	8	21	12	28	12	20	
	3	14	23	19	5	20	12	26	11	20	
	4	15	20	17	5	19	9	25	11	18	
	5	19	20	15	3	19	7	24	10	18	
	6	29	16	13	2	18	3	24	10	17	
18	1	7	24	17	13	21	27	8	18	19	
	2	10	18	16	11	18	24	8	16	18	
	3	13	16	12	7	15	21	6	14	14	
	4	16	16	9	7	13	14	4	12	13	
	5	21	12	8	4	9	10	2	10	12	
	6	27	10	7	2	7	9	1	9	8	
19	1	3	25	19	9	29	10	14	17	25	
	2	8	25	18	8	28	9	13	15	23	
	3	11	21	18	8	27	9	13	13	20	
	4	20	21	18	8	27	8	12	13	19	
	5	26	18	17	8	26	8	10	10	16	
	6	30	16	17	8	26	7	9	7	16	
20	1	3	23	26	24	13	27	21	6	29	
	2	8	21	23	23	12	26	20	6	27	
	3	12	17	19	21	9	26	20	6	25	
	4	14	16	17	18	9	25	17	6	22	
	5	15	11	14	14	6	24	17	6	21	
	6	25	9	11	13	6	24	15	6	20	
21	1	4	27	26	10	13	29	17	10	18	
	2	10	26	25	10	12	28	17	8	14	
	3	11	24	23	10	12	27	15	8	13	
	4	17	23	20	10	11	27	10	7	10	
	5	25	22	17	10	11	26	7	4	9	
	6	26	20	16	10	11	25	5	2	8	
22	1	7	16	29	10	17	13	15	24	15	
	2	9	16	28	9	17	13	14	22	15	
	3	10	16	27	8	17	10	12	17	14	
	4	13	16	25	7	17	6	12	16	14	
	5	18	15	24	7	16	4	11	14	13	
	6	19	15	24	6	16	2	10	9	13	
23	1	1	28	18	26	24	19	8	20	22	
	2	2	23	17	21	24	18	7	18	20	
	3	3	21	17	18	24	17	7	16	19	
	4	7	20	16	17	24	15	6	15	18	
	5	9	16	16	13	24	13	6	12	17	
	6	14	14	16	10	24	12	5	12	14	
24	1	2	28	30	24	23	18	24	30	29	
	2	12	27	23	23	19	18	20	28	21	
	3	14	27	22	23	16	15	17	27	17	
	4	15	27	20	23	15	14	12	25	13	
	5	16	26	15	21	13	11	10	24	12	
	6	26	26	12	21	9	11	6	23	5	
25	1	5	22	19	30	5	13	22	23	5	
	2	9	19	18	22	4	11	20	22	5	
	3	13	19	15	18	3	11	20	21	5	
	4	21	17	13	11	3	9	17	20	5	
	5	23	12	11	6	1	7	14	19	5	
	6	27	11	10	4	1	5	13	19	5	
26	1	9	14	24	4	25	16	23	24	23	
	2	15	14	22	4	25	13	23	20	22	
	3	16	14	20	3	22	13	20	18	21	
	4	19	14	17	3	20	8	18	17	17	
	5	24	14	12	2	18	7	15	11	16	
	6	29	14	12	2	16	2	13	10	13	
27	1	2	28	27	5	10	23	17	22	17	
	2	4	25	25	5	9	22	15	15	15	
	3	5	25	22	5	9	21	14	14	12	
	4	6	22	19	5	9	20	14	11	12	
	5	15	21	13	4	9	19	12	8	11	
	6	22	20	11	4	9	19	12	2	9	
28	1	1	25	19	19	26	16	23	24	26	
	2	13	25	19	15	24	12	20	19	20	
	3	17	24	19	11	21	12	20	15	16	
	4	20	23	19	9	21	9	18	10	14	
	5	23	22	18	5	19	7	17	7	11	
	6	25	21	18	4	17	5	16	1	10	
29	1	5	15	22	28	16	12	8	14	21	
	2	9	12	22	21	16	9	7	14	20	
	3	12	12	18	18	16	8	5	10	19	
	4	25	10	17	14	15	6	4	8	18	
	5	26	8	14	9	15	5	3	7	18	
	6	30	5	11	8	15	3	3	6	17	
30	1	1	20	17	21	19	27	22	30	28	
	2	2	19	16	20	17	25	20	25	26	
	3	11	19	16	19	16	23	15	23	25	
	4	25	19	16	18	16	19	13	20	24	
	5	26	19	15	18	14	17	11	16	24	
	6	28	19	15	17	14	12	9	15	22	
31	1	4	20	27	15	19	28	16	22	28	
	2	5	20	23	14	14	28	13	18	28	
	3	6	20	21	13	13	28	13	16	28	
	4	9	20	18	13	7	27	8	14	28	
	5	14	20	12	11	7	27	5	12	28	
	6	16	20	11	11	4	27	3	12	28	
32	1	3	27	25	27	25	26	15	21	18	
	2	14	24	24	22	22	24	14	16	18	
	3	18	21	22	19	22	23	14	16	16	
	4	19	18	21	17	19	23	14	11	16	
	5	23	11	21	10	15	21	12	7	15	
	6	27	8	20	7	13	21	12	3	14	
33	1	2	4	22	30	25	20	24	24	26	
	2	3	4	20	22	25	19	23	22	22	
	3	4	4	19	22	25	19	23	21	15	
	4	12	4	19	18	25	17	23	21	14	
	5	18	4	17	11	25	16	23	18	8	
	6	21	4	15	6	25	16	23	17	3	
34	1	9	19	23	24	17	27	18	25	23	
	2	17	18	22	24	14	26	17	24	21	
	3	18	16	18	19	13	26	12	24	16	
	4	23	13	17	12	12	26	10	23	11	
	5	24	10	14	10	11	25	6	22	7	
	6	28	7	10	6	11	24	2	22	7	
35	1	6	29	27	27	15	18	12	9	25	
	2	7	21	21	25	13	17	11	8	20	
	3	10	20	16	22	11	17	10	8	20	
	4	15	18	15	20	10	17	10	8	15	
	5	16	14	10	19	6	17	9	8	12	
	6	20	7	3	17	5	17	8	8	11	
36	1	10	12	27	17	26	18	24	8	30	
	2	14	12	23	17	25	17	17	7	24	
	3	17	9	20	17	24	17	14	6	20	
	4	25	7	13	17	23	17	9	5	15	
	5	28	4	11	17	21	17	8	5	11	
	6	29	4	7	17	20	17	3	4	7	
37	1	9	24	26	20	15	12	8	8	25	
	2	23	23	25	18	14	11	7	6	22	
	3	24	20	23	15	13	10	7	6	21	
	4	25	15	17	15	10	10	4	4	20	
	5	26	10	14	12	8	9	3	3	19	
	6	28	9	11	11	8	9	2	3	18	
38	1	2	30	17	22	14	27	26	28	26	
	2	9	28	16	21	14	26	26	25	25	
	3	16	27	12	19	12	26	25	23	25	
	4	21	26	8	19	11	26	23	19	23	
	5	22	25	7	16	10	26	22	19	22	
	6	25	25	4	16	9	26	22	17	22	
39	1	6	9	28	22	15	14	7	21	15	
	2	8	8	25	19	12	12	6	16	14	
	3	10	6	22	17	11	11	5	14	14	
	4	12	4	19	14	10	8	5	10	12	
	5	27	3	17	10	7	5	3	7	11	
	6	30	1	12	6	6	1	3	6	11	
40	1	1	23	28	29	20	6	21	23	12	
	2	6	21	26	24	19	5	18	20	10	
	3	7	16	23	15	19	3	13	19	8	
	4	11	14	23	12	18	2	11	18	8	
	5	20	7	21	8	16	1	6	14	5	
	6	29	2	18	5	15	1	3	14	5	
41	1	1	20	18	28	12	16	19	22	28	
	2	5	20	15	20	11	14	18	20	27	
	3	17	20	15	17	7	13	16	20	26	
	4	21	19	13	15	6	12	15	16	24	
	5	26	19	10	9	4	12	15	15	24	
	6	27	19	9	4	2	11	13	13	23	
42	1	5	24	24	28	20	28	22	19	16	
	2	6	22	21	22	17	25	22	18	13	
	3	14	17	20	21	15	24	22	17	11	
	4	16	13	17	15	11	20	22	16	9	
	5	20	10	12	9	8	18	22	16	9	
	6	27	10	9	6	6	18	22	15	5	
43	1	4	6	23	11	16	21	28	25	28	
	2	10	5	21	10	15	21	28	23	26	
	3	13	5	17	9	15	21	28	21	22	
	4	15	5	14	9	14	21	27	19	20	
	5	21	5	13	7	14	21	27	15	19	
	6	28	5	8	7	13	21	26	13	16	
44	1	9	14	21	20	29	18	22	26	20	
	2	10	13	19	20	24	16	22	25	19	
	3	11	11	18	20	21	14	22	25	15	
	4	18	9	17	20	21	12	22	25	12	
	5	20	6	13	20	17	12	22	24	10	
	6	23	5	12	20	13	10	22	23	9	
45	1	1	22	7	21	27	28	22	13	12	
	2	5	21	5	17	22	28	20	12	11	
	3	6	17	5	15	19	28	18	12	8	
	4	8	16	3	13	13	27	16	11	5	
	5	17	12	3	11	7	27	13	10	5	
	6	24	11	2	6	6	27	10	10	2	
46	1	8	27	28	28	18	8	23	8	18	
	2	16	26	22	27	16	8	22	8	17	
	3	18	25	21	27	14	8	21	6	17	
	4	20	25	16	27	12	8	19	5	17	
	5	21	25	12	26	10	8	18	5	17	
	6	28	24	3	26	8	8	16	3	17	
47	1	4	12	22	12	19	16	20	15	29	
	2	5	11	20	11	15	15	17	13	27	
	3	13	11	17	10	12	14	15	9	23	
	4	23	11	14	9	10	14	11	8	22	
	5	25	11	9	8	6	12	11	6	19	
	6	30	11	7	8	2	10	9	4	18	
48	1	1	29	18	26	8	25	29	26	12	
	2	7	28	18	25	7	23	21	24	10	
	3	15	26	18	24	7	19	17	19	6	
	4	17	25	17	22	7	17	15	16	5	
	5	19	24	17	21	7	14	6	12	3	
	6	21	24	17	20	7	12	2	11	1	
49	1	14	29	19	19	9	16	10	26	25	
	2	20	25	18	15	8	15	9	25	18	
	3	23	23	18	12	8	15	9	21	16	
	4	24	22	18	10	7	14	9	20	13	
	5	27	18	18	8	6	14	8	15	9	
	6	30	16	18	3	6	14	8	14	6	
50	1	5	15	14	24	23	24	28	4	16	
	2	13	13	12	24	21	18	27	4	14	
	3	17	12	12	23	20	16	25	4	12	
	4	21	7	11	23	19	14	22	3	11	
	5	24	6	11	22	16	11	19	3	9	
	6	28	4	10	21	14	6	17	2	6	
51	1	3	13	27	25	25	12	17	21	12	
	2	11	11	26	25	22	12	17	19	12	
	3	14	8	25	25	17	12	16	16	12	
	4	15	7	24	25	16	12	16	13	12	
	5	23	3	24	25	11	12	15	8	12	
	6	30	2	23	25	6	12	15	5	12	
52	1	0	0	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	R 3	R 4	N 1	N 2	N 3	N 4
	58	64	64	56	860	876	866	950

************************************************************************
