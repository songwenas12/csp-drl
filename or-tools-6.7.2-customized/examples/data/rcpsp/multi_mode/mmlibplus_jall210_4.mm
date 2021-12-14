jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 4 R
- nonrenewable              : 4 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	7		2 3 4 5 6 10 11 
2	6	4		16 15 13 7 
3	6	6		24 19 17 16 13 9 
4	6	4		15 14 12 8 
5	6	4		24 17 14 12 
6	6	4		24 19 17 14 
7	6	2		12 8 
8	6	4		23 20 18 17 
9	6	4		26 22 20 14 
10	6	3		23 16 13 
11	6	6		26 24 23 22 21 20 
12	6	5		26 25 22 21 20 
13	6	4		27 26 22 18 
14	6	3		27 23 18 
15	6	3		24 23 22 
16	6	4		30 27 26 25 
17	6	3		27 26 22 
18	6	2		25 21 
19	6	2		25 21 
20	6	5		36 30 29 28 27 
21	6	5		37 36 32 30 28 
22	6	4		37 30 29 28 
23	6	2		36 25 
24	6	3		30 29 28 
25	6	2		29 28 
26	6	3		36 33 28 
27	6	4		37 35 32 31 
28	6	3		38 35 31 
29	6	3		39 38 32 
30	6	3		38 34 33 
31	6	4		42 40 39 34 
32	6	2		34 33 
33	6	3		44 42 40 
34	6	3		46 44 41 
35	6	2		46 41 
36	6	2		44 41 
37	6	2		46 41 
38	6	2		46 42 
39	6	5		51 50 48 45 43 
40	6	4		51 46 45 43 
41	6	4		51 50 45 43 
42	6	4		50 48 45 43 
43	6	2		49 47 
44	6	2		48 47 
45	6	1		47 
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
2	1	10	17	28	6	26	27	26	18	6	
	2	14	17	23	5	22	21	24	16	6	
	3	18	17	17	4	19	20	19	16	6	
	4	21	16	14	2	16	14	17	15	6	
	5	28	15	11	1	16	10	13	15	6	
	6	30	15	7	1	10	8	11	14	6	
3	1	6	25	10	14	12	27	21	18	26	
	2	12	24	9	11	11	25	19	14	25	
	3	13	24	9	11	11	24	17	13	25	
	4	16	23	9	10	11	19	17	8	23	
	5	25	22	7	8	11	16	15	7	23	
	6	28	22	7	7	11	14	13	5	22	
4	1	3	15	15	27	15	29	27	17	14	
	2	4	14	14	24	13	25	26	12	13	
	3	12	13	13	21	12	24	26	12	9	
	4	14	13	12	17	10	20	26	10	8	
	5	22	13	12	17	9	19	26	8	4	
	6	29	12	11	13	6	16	26	6	3	
5	1	20	28	24	17	22	17	28	15	17	
	2	21	22	23	15	22	16	20	14	13	
	3	22	21	22	14	20	16	17	13	11	
	4	23	17	21	13	18	16	13	11	7	
	5	25	12	19	13	17	16	9	8	4	
	6	28	10	19	12	16	16	3	6	3	
6	1	1	27	11	18	12	12	13	8	17	
	2	3	22	11	15	11	11	9	7	15	
	3	10	20	11	14	10	11	7	6	14	
	4	14	17	11	10	10	11	5	6	12	
	5	21	15	11	9	8	10	3	4	11	
	6	23	9	11	6	6	9	3	2	9	
7	1	5	27	16	5	12	22	26	22	21	
	2	6	26	16	5	10	18	25	19	18	
	3	20	25	15	5	9	18	21	17	15	
	4	21	25	13	5	8	15	21	16	15	
	5	22	25	12	5	8	11	19	16	13	
	6	24	24	11	5	7	9	16	14	11	
8	1	3	20	16	4	28	29	20	18	9	
	2	8	20	14	3	27	27	19	13	8	
	3	17	20	14	3	27	24	19	12	8	
	4	19	19	12	2	27	23	17	7	7	
	5	27	19	12	1	27	22	15	6	6	
	6	28	19	11	1	27	20	15	2	6	
9	1	14	28	26	28	23	20	2	22	14	
	2	15	28	23	27	23	19	2	21	14	
	3	17	27	18	26	21	18	2	21	13	
	4	20	26	14	24	20	17	1	19	13	
	5	25	25	12	21	19	16	1	19	12	
	6	26	23	7	20	19	15	1	17	12	
10	1	5	25	23	27	28	9	16	17	8	
	2	12	24	18	26	26	8	15	13	7	
	3	22	24	15	20	24	6	10	11	6	
	4	26	24	10	18	22	5	10	10	4	
	5	29	24	8	14	21	4	7	7	3	
	6	30	24	3	11	19	3	2	6	3	
11	1	2	17	18	23	20	19	26	25	23	
	2	3	14	17	22	20	15	25	22	21	
	3	12	11	16	22	15	15	17	19	20	
	4	21	8	15	21	12	10	17	17	19	
	5	23	5	14	21	9	8	10	12	15	
	6	24	2	14	21	4	3	7	9	14	
12	1	6	24	28	14	15	20	7	27	6	
	2	14	21	25	14	14	20	6	19	5	
	3	15	19	23	12	14	20	6	15	5	
	4	19	13	21	9	14	20	6	11	3	
	5	25	9	17	6	13	19	6	5	3	
	6	27	5	12	5	13	19	6	1	2	
13	1	4	15	19	27	23	21	26	30	20	
	2	8	13	19	27	21	19	25	27	19	
	3	9	12	19	27	21	18	22	26	17	
	4	14	12	18	27	20	17	19	26	17	
	5	23	11	18	26	19	16	18	23	15	
	6	28	10	18	26	17	15	14	23	14	
14	1	2	15	3	16	12	26	9	19	17	
	2	13	13	3	16	11	20	9	18	16	
	3	18	11	3	16	11	16	8	17	16	
	4	21	11	2	15	11	15	7	17	14	
	5	22	8	2	15	11	10	7	17	14	
	6	30	6	1	15	11	10	6	16	13	
15	1	2	29	12	7	19	10	25	10	14	
	2	12	27	10	5	17	9	23	10	14	
	3	15	27	9	5	15	9	23	10	14	
	4	20	27	7	4	12	9	19	10	14	
	5	22	25	6	3	11	8	17	10	13	
	6	27	25	4	1	9	8	17	10	13	
16	1	4	15	20	11	18	12	25	16	14	
	2	15	13	19	9	17	9	24	15	11	
	3	16	10	19	9	16	8	24	10	9	
	4	18	9	18	5	16	7	23	8	5	
	5	19	4	16	3	16	6	21	6	3	
	6	29	4	14	3	15	4	21	4	3	
17	1	2	4	21	25	28	25	11	11	29	
	2	3	4	18	25	27	23	11	10	29	
	3	10	4	18	18	26	19	11	10	28	
	4	15	3	15	16	24	17	11	10	28	
	5	16	3	13	13	23	13	11	10	27	
	6	26	2	13	8	22	11	11	10	26	
18	1	4	7	10	19	9	25	24	28	17	
	2	5	7	9	18	9	22	21	25	17	
	3	9	7	7	17	8	19	17	25	17	
	4	18	6	6	17	7	12	15	21	17	
	5	29	6	5	16	7	11	9	19	17	
	6	30	6	4	15	6	5	7	18	17	
19	1	4	24	29	17	25	25	11	22	7	
	2	5	22	26	15	24	25	10	19	7	
	3	7	18	25	14	24	25	10	14	6	
	4	10	17	24	13	22	24	9	12	5	
	5	21	14	23	13	22	23	7	8	3	
	6	22	12	23	12	21	23	7	5	3	
20	1	5	19	18	14	23	24	12	27	6	
	2	6	19	18	13	21	22	11	24	6	
	3	8	19	18	13	18	20	10	22	4	
	4	9	19	18	13	18	20	10	16	3	
	5	20	19	18	13	16	17	9	13	2	
	6	29	19	18	13	14	16	8	13	1	
21	1	4	8	15	22	24	29	25	11	9	
	2	15	7	14	22	20	29	24	11	9	
	3	18	7	14	22	19	29	24	11	9	
	4	19	6	12	22	16	28	22	11	9	
	5	25	4	11	21	12	28	22	10	9	
	6	30	3	11	21	11	27	21	10	9	
22	1	8	23	17	17	17	28	12	27	27	
	2	18	20	17	17	16	26	11	27	25	
	3	19	18	15	11	16	24	11	26	20	
	4	25	16	14	7	15	24	10	26	20	
	5	26	14	12	4	15	22	8	25	16	
	6	27	11	11	1	14	20	8	25	13	
23	1	2	29	27	19	26	28	9	24	15	
	2	4	25	26	19	23	24	9	22	14	
	3	7	24	25	19	21	22	9	22	12	
	4	12	21	22	19	18	20	8	20	11	
	5	14	19	21	19	16	17	7	18	11	
	6	17	16	20	19	12	13	7	16	10	
24	1	2	3	11	3	25	27	8	22	22	
	2	4	3	11	3	23	24	8	20	20	
	3	11	3	7	3	21	21	8	17	19	
	4	19	2	5	3	20	21	7	15	17	
	5	20	2	4	2	14	16	6	12	16	
	6	29	2	3	2	12	13	6	8	13	
25	1	7	30	22	16	10	19	26	26	20	
	2	9	27	21	13	10	16	25	21	19	
	3	18	26	21	11	8	14	25	18	17	
	4	27	22	19	7	8	11	23	16	16	
	5	28	21	18	4	7	7	20	12	15	
	6	29	19	18	3	5	6	20	6	14	
26	1	2	22	21	23	11	20	19	26	19	
	2	3	20	16	21	11	16	19	23	19	
	3	4	17	16	18	9	16	19	19	18	
	4	20	13	9	14	8	13	19	15	17	
	5	27	7	9	13	7	11	19	7	15	
	6	28	4	4	10	6	6	19	3	15	
27	1	3	21	6	26	23	24	24	22	8	
	2	16	18	6	25	21	24	24	22	7	
	3	22	18	5	23	21	20	24	22	6	
	4	23	17	3	21	18	16	24	22	6	
	5	25	16	2	18	17	12	23	22	5	
	6	26	15	2	16	17	12	23	22	5	
28	1	8	13	18	27	12	28	13	24	17	
	2	10	13	16	27	9	27	10	23	17	
	3	11	11	13	27	9	25	10	22	15	
	4	20	9	12	27	6	24	8	22	12	
	5	28	8	9	26	5	24	4	20	11	
	6	29	5	8	26	3	22	4	20	8	
29	1	4	14	6	15	25	28	27	23	8	
	2	7	12	5	15	23	25	27	19	8	
	3	8	11	5	14	21	20	23	15	7	
	4	24	10	4	12	19	15	21	14	6	
	5	28	9	4	11	16	13	20	7	5	
	6	29	9	3	10	14	8	19	6	5	
30	1	3	28	21	20	24	27	6	25	23	
	2	4	21	16	13	23	22	5	23	23	
	3	5	19	16	12	21	18	5	23	17	
	4	21	14	12	9	19	15	4	22	11	
	5	22	11	12	4	17	12	4	22	9	
	6	25	11	8	1	16	10	4	21	6	
31	1	4	21	26	21	4	23	29	18	17	
	2	13	18	23	18	4	22	29	16	16	
	3	14	16	21	13	4	22	29	11	11	
	4	16	12	20	12	3	22	29	9	8	
	5	19	7	16	7	3	21	29	6	6	
	6	21	4	12	7	2	21	29	5	2	
32	1	4	20	24	10	28	29	21	10	8	
	2	5	19	23	9	20	22	20	8	8	
	3	10	15	21	9	15	16	18	7	6	
	4	14	8	20	9	11	13	18	5	6	
	5	22	8	20	9	9	8	16	4	4	
	6	30	3	17	9	3	5	16	4	2	
33	1	5	21	14	16	24	16	28	29	14	
	2	20	20	14	12	19	14	27	26	13	
	3	21	20	13	10	19	13	27	23	12	
	4	22	19	11	9	13	12	26	21	12	
	5	25	19	10	7	12	11	25	21	10	
	6	28	19	10	4	7	11	25	18	10	
34	1	3	14	20	26	7	29	29	27	18	
	2	15	13	14	26	6	22	25	22	18	
	3	16	13	12	22	6	17	21	21	15	
	4	23	12	11	22	6	15	19	14	14	
	5	24	11	6	20	5	11	18	10	11	
	6	30	11	5	17	5	7	15	7	9	
35	1	1	28	19	18	21	10	20	28	23	
	2	2	28	19	18	19	8	20	27	20	
	3	26	28	18	13	14	8	20	25	20	
	4	28	28	18	9	11	5	20	25	18	
	5	29	27	18	5	10	5	20	24	17	
	6	30	27	17	2	6	4	20	23	16	
36	1	5	13	17	18	9	18	7	6	26	
	2	6	11	14	17	9	17	6	6	25	
	3	9	11	12	17	7	17	6	6	22	
	4	12	10	10	17	6	16	5	6	17	
	5	16	9	8	17	5	16	5	6	16	
	6	24	8	5	17	3	16	4	6	12	
37	1	15	25	29	20	23	26	26	14	29	
	2	16	22	24	18	21	21	26	14	27	
	3	18	22	24	18	20	16	25	14	27	
	4	19	16	22	14	18	13	25	14	27	
	5	22	13	20	10	18	7	24	14	26	
	6	23	12	18	8	16	6	24	14	25	
38	1	7	29	18	24	18	13	13	17	16	
	2	11	26	16	22	17	12	12	15	13	
	3	13	26	14	22	12	10	12	15	12	
	4	18	25	13	19	12	10	11	15	8	
	5	21	23	11	17	8	9	11	14	7	
	6	23	23	9	16	6	8	10	13	4	
39	1	1	22	3	27	12	28	17	9	18	
	2	7	20	2	26	12	26	15	9	15	
	3	15	17	2	25	11	23	14	8	13	
	4	22	14	2	23	11	22	13	8	8	
	5	24	13	2	22	10	18	11	7	7	
	6	27	7	2	21	9	16	8	7	3	
40	1	8	27	29	24	19	28	16	18	25	
	2	9	25	28	22	18	27	14	17	23	
	3	15	22	28	20	17	27	11	17	21	
	4	17	19	28	17	16	25	11	16	14	
	5	29	17	28	15	15	25	5	15	12	
	6	30	14	28	14	15	23	4	15	5	
41	1	1	25	24	28	28	15	9	20	13	
	2	3	22	19	26	27	15	8	17	12	
	3	16	19	17	24	25	14	7	14	11	
	4	17	17	10	23	20	11	7	11	9	
	5	19	15	8	20	19	11	6	10	9	
	6	23	15	5	19	18	10	6	7	8	
42	1	2	26	22	19	24	23	26	16	20	
	2	10	22	22	18	20	22	23	15	16	
	3	11	18	20	14	15	19	22	10	15	
	4	19	14	20	13	10	18	18	9	8	
	5	20	11	18	8	8	15	18	6	8	
	6	25	10	18	6	2	14	14	3	4	
43	1	5	15	28	22	18	23	23	17	9	
	2	7	15	24	22	17	22	22	17	9	
	3	13	14	16	20	16	22	22	17	8	
	4	17	12	15	20	16	22	22	17	6	
	5	23	11	11	18	14	22	21	17	5	
	6	28	10	3	17	14	22	20	17	5	
44	1	8	4	27	19	19	14	25	17	11	
	2	9	4	22	18	15	14	24	15	9	
	3	20	4	20	17	15	12	23	13	6	
	4	21	3	14	16	10	11	18	13	5	
	5	22	2	10	15	8	11	18	10	2	
	6	27	2	8	15	8	10	14	9	2	
45	1	4	21	22	21	22	7	25	27	23	
	2	5	20	19	20	20	6	23	27	22	
	3	11	19	18	17	16	4	21	27	22	
	4	13	19	16	16	11	4	20	27	22	
	5	25	17	16	13	10	2	20	27	21	
	6	27	17	14	11	8	2	19	27	21	
46	1	2	20	14	14	19	13	27	20	16	
	2	3	18	14	14	19	11	21	19	15	
	3	14	18	14	11	18	10	18	19	15	
	4	15	17	14	10	18	10	16	18	13	
	5	18	16	14	8	18	8	12	16	11	
	6	23	15	14	4	17	8	8	16	10	
47	1	1	25	22	25	24	29	17	24	16	
	2	13	25	21	21	24	27	14	19	16	
	3	15	21	21	15	24	27	11	18	16	
	4	18	20	20	10	24	26	9	13	16	
	5	19	19	20	5	24	25	9	5	16	
	6	25	16	19	4	24	25	4	5	16	
48	1	1	15	17	10	20	20	25	10	19	
	2	4	14	17	10	15	19	23	9	16	
	3	13	14	15	10	15	19	21	7	13	
	4	14	12	15	10	11	19	19	7	12	
	5	15	12	14	10	9	17	13	6	10	
	6	30	11	13	10	7	17	13	3	6	
49	1	4	26	5	27	25	14	21	5	12	
	2	16	25	5	23	24	14	18	5	11	
	3	21	22	5	23	23	14	17	5	10	
	4	25	21	5	17	23	14	14	4	10	
	5	26	18	4	17	23	14	14	4	8	
	6	28	17	4	13	22	14	10	3	8	
50	1	8	14	24	20	5	21	24	23	19	
	2	9	13	23	19	5	21	21	19	13	
	3	11	13	22	17	5	21	20	16	11	
	4	18	13	21	16	5	21	14	13	10	
	5	20	13	19	16	5	21	11	10	7	
	6	29	13	19	15	5	21	11	9	4	
51	1	6	28	17	24	20	23	17	22	26	
	2	10	26	14	23	18	18	16	20	22	
	3	11	20	10	22	15	15	15	20	21	
	4	24	17	10	19	13	14	15	18	16	
	5	25	14	4	18	8	10	14	17	12	
	6	30	11	2	15	5	5	13	16	11	
52	1	0	0	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	R 3	R 4	N 1	N 2	N 3	N 4
	61	54	54	54	965	879	869	738

************************************************************************
