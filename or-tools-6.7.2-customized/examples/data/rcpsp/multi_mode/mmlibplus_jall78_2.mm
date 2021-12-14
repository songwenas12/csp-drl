jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 4 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	5		2 3 5 6 9 
2	3	3		11 7 4 
3	3	2		10 8 
4	3	4		18 17 14 10 
5	3	3		18 12 10 
6	3	5		19 18 16 15 14 
7	3	2		22 10 
8	3	2		16 11 
9	3	6		22 20 19 16 15 13 
10	3	5		20 19 16 15 13 
11	3	4		24 21 17 14 
12	3	2		22 15 
13	3	6		30 29 26 25 24 23 
14	3	5		30 26 25 23 20 
15	3	7		35 30 29 27 26 25 24 
16	3	6		35 29 27 25 24 21 
17	3	4		26 25 22 19 
18	3	4		35 32 25 22 
19	3	5		35 34 32 30 27 
20	3	4		35 34 29 27 
21	3	4		34 32 30 28 
22	3	3		34 30 27 
23	3	3		35 34 27 
24	3	4		41 34 32 28 
25	3	3		41 34 28 
26	3	3		41 34 28 
27	3	2		41 28 
28	3	3		39 37 31 
29	3	4		46 39 36 32 
30	3	6		46 41 40 39 37 33 
31	3	4		51 46 36 33 
32	3	3		51 40 33 
33	3	3		45 44 38 
34	3	3		46 39 38 
35	3	3		46 41 39 
36	3	3		45 42 40 
37	3	1		38 
38	3	3		50 43 42 
39	3	3		50 43 42 
40	3	3		50 49 43 
41	3	3		50 48 42 
42	3	2		49 47 
43	3	2		48 47 
44	3	2		49 47 
45	3	1		48 
46	3	1		47 
47	3	1		52 
48	3	1		52 
49	3	1		52 
50	3	1		52 
51	3	1		52 
52	1	0		
************************************************************************
REQUESTS/DURATIONS
jobnr.	mode	dur	R1	R2	R3	R4	N1	N2	
------------------------------------------------------------------------
1	1	0	0	0	0	0	0	0	
2	1	1	14	16	20	20	24	24	
	2	15	13	16	12	19	15	15	
	3	19	13	16	7	17	5	8	
3	1	13	27	22	26	16	27	17	
	2	16	21	21	23	14	26	15	
	3	26	18	20	21	12	23	10	
4	1	11	12	27	24	16	15	6	
	2	12	9	12	22	8	13	6	
	3	25	7	2	20	6	11	4	
5	1	14	18	9	13	20	19	6	
	2	21	13	8	9	19	13	6	
	3	23	4	7	9	19	13	5	
6	1	4	25	11	12	21	5	23	
	2	19	22	8	10	18	5	20	
	3	23	20	7	9	14	4	18	
7	1	1	17	22	14	23	19	29	
	2	2	16	20	11	18	18	26	
	3	20	16	19	4	11	11	24	
8	1	4	9	17	24	25	23	23	
	2	14	7	14	15	24	15	16	
	3	30	1	6	14	23	10	11	
9	1	18	18	16	19	26	26	30	
	2	23	10	13	16	25	19	28	
	3	26	4	11	14	24	10	28	
10	1	11	18	10	21	12	24	21	
	2	22	18	9	15	11	14	18	
	3	23	18	6	12	8	4	17	
11	1	4	19	17	19	18	20	29	
	2	22	17	15	16	14	17	29	
	3	28	15	10	15	4	9	29	
12	1	11	26	20	10	24	20	12	
	2	27	22	20	8	22	15	8	
	3	28	18	19	3	18	11	6	
13	1	14	22	13	24	20	14	29	
	2	17	12	10	20	17	9	28	
	3	30	9	7	20	16	4	28	
14	1	3	30	24	27	27	12	20	
	2	20	30	18	27	21	8	17	
	3	25	30	13	23	9	6	11	
15	1	3	19	15	27	23	18	13	
	2	10	19	13	22	21	12	13	
	3	26	8	12	14	19	8	9	
16	1	5	1	19	10	13	24	5	
	2	13	1	14	9	13	17	5	
	3	20	1	9	6	13	13	5	
17	1	1	19	19	8	19	13	9	
	2	17	17	17	8	11	12	8	
	3	21	11	17	1	10	5	7	
18	1	11	7	20	21	18	18	23	
	2	25	6	18	19	17	16	18	
	3	28	5	11	14	17	15	14	
19	1	6	26	27	12	21	23	16	
	2	23	25	17	12	19	21	14	
	3	30	25	12	11	18	20	11	
20	1	10	23	29	19	12	27	22	
	2	16	16	28	19	10	27	10	
	3	27	15	27	19	9	27	7	
21	1	4	18	20	11	25	15	28	
	2	12	14	18	9	18	15	28	
	3	19	10	13	6	13	15	25	
22	1	4	15	18	10	25	13	20	
	2	8	10	17	9	25	6	16	
	3	17	6	5	9	23	1	10	
23	1	22	12	11	7	26	20	19	
	2	23	11	10	7	20	10	16	
	3	28	9	8	5	12	9	9	
24	1	4	22	9	21	20	6	19	
	2	5	16	9	15	15	5	18	
	3	13	11	4	13	15	4	18	
25	1	6	21	22	21	9	28	27	
	2	7	19	21	15	8	20	22	
	3	15	13	21	10	6	19	20	
26	1	1	25	26	21	12	22	7	
	2	8	23	22	14	6	14	7	
	3	17	22	18	14	6	9	7	
27	1	4	11	19	25	22	27	30	
	2	17	6	18	19	19	27	30	
	3	24	3	18	8	12	27	30	
28	1	16	18	29	16	7	28	27	
	2	17	13	28	13	7	16	20	
	3	20	8	28	11	7	9	19	
29	1	1	28	28	16	14	8	29	
	2	11	16	17	8	13	6	21	
	3	14	11	9	4	13	4	16	
30	1	2	17	6	11	23	30	14	
	2	5	9	5	7	22	29	13	
	3	30	2	2	6	22	27	8	
31	1	3	23	18	22	18	25	27	
	2	11	21	9	17	18	23	17	
	3	18	19	6	9	14	21	3	
32	1	11	21	16	27	11	17	12	
	2	18	15	11	27	7	15	12	
	3	30	10	6	26	5	15	11	
33	1	9	22	22	17	25	12	12	
	2	10	19	22	16	21	12	12	
	3	14	15	19	16	20	10	12	
34	1	8	12	28	27	14	16	23	
	2	9	7	25	23	14	16	18	
	3	15	1	23	22	14	14	12	
35	1	12	26	21	23	24	21	17	
	2	20	25	11	21	22	18	9	
	3	27	25	10	18	18	15	2	
36	1	1	21	11	25	26	15	24	
	2	3	20	10	17	24	10	22	
	3	29	5	10	8	23	6	15	
37	1	2	13	24	26	26	4	23	
	2	23	11	21	24	15	3	20	
	3	26	8	16	21	3	3	16	
38	1	6	15	20	22	10	19	6	
	2	18	15	14	18	10	15	4	
	3	30	13	7	18	10	9	4	
39	1	20	15	19	27	25	21	26	
	2	25	11	15	19	19	15	21	
	3	29	10	14	13	17	11	21	
40	1	11	26	11	19	19	21	22	
	2	13	25	11	14	17	16	14	
	3	26	23	8	13	10	15	9	
41	1	10	10	20	8	12	18	28	
	2	19	7	18	7	8	18	11	
	3	28	3	14	4	7	13	4	
42	1	16	15	10	8	22	18	25	
	2	24	12	9	6	19	13	14	
	3	30	8	8	4	18	2	5	
43	1	8	20	10	17	19	20	17	
	2	9	20	4	14	18	18	15	
	3	21	20	2	13	13	18	15	
44	1	4	17	15	22	27	10	23	
	2	6	10	10	16	16	8	20	
	3	15	6	9	12	10	5	18	
45	1	8	26	16	18	26	21	29	
	2	16	20	11	16	16	20	23	
	3	17	16	10	6	12	19	22	
46	1	4	13	11	18	21	25	4	
	2	5	9	10	11	18	20	3	
	3	12	7	9	2	7	13	3	
47	1	13	26	20	24	21	25	17	
	2	23	20	18	18	18	18	14	
	3	27	17	16	14	14	13	6	
48	1	13	12	10	7	26	8	21	
	2	22	11	8	6	15	6	14	
	3	25	10	3	6	7	6	4	
49	1	10	14	18	25	20	22	5	
	2	12	9	14	12	15	20	5	
	3	13	7	13	6	9	17	5	
50	1	3	6	11	29	29	19	16	
	2	14	5	8	25	23	16	8	
	3	27	4	5	17	21	8	4	
51	1	2	15	17	5	12	9	13	
	2	5	7	13	4	11	7	8	
	3	7	5	11	3	7	7	8	
52	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	R 3	R 4	N 1	N 2
	77	75	68	75	843	878

************************************************************************
