jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 4 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	12		2 3 4 6 7 8 9 10 11 12 13 17 
2	6	4		18 15 14 5 
3	6	7		51 35 24 22 18 16 14 
4	6	9		51 34 32 25 24 22 20 19 18 
5	6	9		51 34 28 26 25 24 22 21 20 
6	6	9		32 28 26 25 24 23 22 21 20 
7	6	8		34 32 26 25 24 21 20 18 
8	6	5		51 34 24 20 14 
9	6	6		32 27 25 22 18 15 
10	6	5		51 24 21 18 15 
11	6	7		51 28 26 25 22 21 20 
12	6	8		34 32 31 28 26 25 24 23 
13	6	8		35 32 31 29 28 27 26 25 
14	6	6		32 31 30 28 27 19 
15	6	6		36 31 30 28 26 20 
16	6	7		49 48 36 34 32 31 25 
17	6	3		40 27 20 
18	6	7		50 47 39 36 31 30 28 
19	6	7		49 46 45 41 36 29 26 
20	6	7		50 46 44 41 39 35 29 
21	6	6		50 47 43 33 31 27 
22	6	9		49 48 47 46 44 43 36 33 31 
23	6	6		49 47 46 43 40 27 
24	6	5		50 49 46 45 27 
25	6	7		50 47 45 44 41 39 30 
26	6	6		48 47 44 43 39 33 
27	6	5		48 44 39 38 36 
28	6	5		49 46 45 44 37 
29	6	3		47 43 33 
30	6	3		46 43 38 
31	6	2		40 38 
32	6	3		50 43 42 
33	6	2		38 37 
34	6	2		43 37 
35	6	2		49 48 
36	6	1		37 
37	6	1		42 
38	6	1		42 
39	6	1		42 
40	6	1		45 
41	6	1		42 
42	6	1		52 
43	6	1		52 
44	6	1		52 
45	6	1		52 
46	6	1		52 
47	6	1		52 
48	6	1		52 
49	6	1		52 
50	6	1		52 
51	6	1		52 
52	1	0		
************************************************************************
REQUESTS/DURATIONS
jobnr.	mode	dur	R1	R2	R3	R4	N1	N2	
------------------------------------------------------------------------
1	1	0	0	0	0	0	0	0	
2	1	5	4	5	3	3	16	21	
	2	6	3	4	3	3	15	20	
	3	7	3	4	3	3	14	20	
	4	21	3	4	3	3	13	19	
	5	26	2	3	3	2	11	19	
	6	30	1	3	3	2	10	18	
3	1	11	4	5	4	3	27	12	
	2	14	4	4	3	3	20	11	
	3	20	4	4	3	3	17	10	
	4	22	4	4	2	2	15	10	
	5	27	4	3	2	2	8	9	
	6	28	4	3	2	2	4	9	
4	1	7	4	1	2	4	29	14	
	2	8	3	1	2	4	26	13	
	3	10	3	1	2	4	26	12	
	4	17	3	1	1	3	25	12	
	5	20	2	1	1	3	24	10	
	6	24	2	1	1	3	23	9	
5	1	9	3	4	5	2	14	27	
	2	12	3	3	5	2	12	27	
	3	14	3	3	5	2	9	25	
	4	16	3	3	5	2	7	24	
	5	27	3	2	5	2	7	24	
	6	28	3	2	5	2	4	23	
6	1	9	5	4	2	4	10	19	
	2	17	4	3	1	4	10	18	
	3	24	3	3	1	4	9	16	
	4	26	3	2	1	4	9	15	
	5	28	2	1	1	3	9	12	
	6	30	2	1	1	3	8	12	
7	1	7	5	4	2	3	24	15	
	2	8	4	4	2	2	23	13	
	3	17	3	4	2	2	23	13	
	4	22	3	3	2	2	21	11	
	5	25	2	3	1	2	21	11	
	6	28	2	3	1	2	20	10	
8	1	3	5	4	5	3	19	4	
	2	5	4	3	4	2	15	3	
	3	8	4	2	4	2	13	3	
	4	15	3	2	4	2	11	2	
	5	23	3	2	3	1	6	1	
	6	30	3	1	3	1	6	1	
9	1	2	3	3	4	3	26	23	
	2	14	2	2	4	3	26	22	
	3	15	2	2	3	3	24	21	
	4	22	2	2	3	3	21	20	
	5	28	2	2	3	3	19	20	
	6	30	2	2	2	3	16	19	
10	1	17	1	4	5	3	28	30	
	2	19	1	3	4	2	27	30	
	3	23	1	3	3	2	27	30	
	4	25	1	3	2	2	27	30	
	5	25	1	1	1	2	25	31	
	6	26	1	1	1	2	25	30	
11	1	14	4	4	4	4	24	24	
	2	15	4	4	4	3	24	21	
	3	23	4	4	3	3	24	20	
	4	24	4	4	3	2	24	17	
	5	25	4	4	3	2	24	16	
	6	29	4	4	2	2	24	14	
12	1	2	4	2	1	3	16	22	
	2	7	4	2	1	3	15	20	
	3	14	3	2	1	3	13	20	
	4	15	3	2	1	3	13	19	
	5	16	1	2	1	3	11	18	
	6	18	1	2	1	3	11	17	
13	1	3	5	5	2	4	28	6	
	2	4	4	3	2	4	27	5	
	3	6	4	3	2	3	27	4	
	4	13	3	3	2	3	24	3	
	5	21	3	2	2	2	23	2	
	6	24	3	1	2	2	21	2	
14	1	1	3	4	2	5	19	24	
	2	6	3	4	2	4	18	24	
	3	12	3	3	2	4	17	23	
	4	17	3	3	2	4	15	21	
	5	24	3	2	2	4	14	21	
	6	28	3	2	2	4	13	19	
15	1	12	5	2	4	1	19	26	
	2	13	4	2	4	1	17	24	
	3	14	4	2	4	1	17	23	
	4	24	4	2	4	1	15	22	
	5	25	4	2	4	1	12	21	
	6	27	4	2	4	1	12	20	
16	1	6	4	1	4	3	7	14	
	2	12	4	1	4	2	7	13	
	3	15	4	1	4	2	6	9	
	4	17	3	1	3	2	6	8	
	5	24	2	1	3	2	6	6	
	6	29	2	1	3	2	5	3	
17	1	1	3	2	4	4	24	21	
	2	2	3	2	4	4	24	20	
	3	20	3	2	4	4	21	17	
	4	25	3	1	4	4	17	16	
	5	29	3	1	4	4	16	15	
	6	30	3	1	4	4	13	14	
18	1	8	4	4	5	3	28	22	
	2	18	4	4	4	3	27	22	
	3	22	3	3	4	3	27	22	
	4	24	2	3	3	3	27	22	
	5	25	2	3	3	3	27	21	
	6	26	1	2	3	3	27	22	
19	1	4	4	4	2	2	22	28	
	2	8	4	4	2	2	19	28	
	3	9	3	3	2	2	17	28	
	4	10	3	2	2	2	16	28	
	5	28	1	1	1	2	14	28	
	6	29	1	1	1	2	12	28	
20	1	14	1	4	2	2	19	24	
	2	15	1	4	1	1	15	21	
	3	16	1	3	1	1	11	21	
	4	17	1	3	1	1	9	19	
	5	21	1	1	1	1	8	16	
	6	22	1	1	1	1	3	16	
21	1	1	3	4	5	2	24	11	
	2	2	2	4	4	2	24	10	
	3	4	2	4	4	2	24	9	
	4	13	1	4	4	1	24	10	
	5	20	1	4	4	1	24	9	
	6	24	1	4	4	1	24	8	
22	1	1	3	2	4	4	24	29	
	2	5	3	1	3	4	19	23	
	3	15	3	1	3	4	17	18	
	4	16	3	1	2	4	15	16	
	5	20	3	1	2	4	9	9	
	6	21	3	1	2	4	4	5	
23	1	11	4	3	4	2	27	4	
	2	12	4	3	4	2	27	3	
	3	13	4	3	3	2	26	2	
	4	16	3	3	2	2	26	2	
	5	24	3	3	1	1	24	2	
	6	30	3	3	1	1	24	1	
24	1	1	4	4	5	1	6	3	
	2	2	4	3	4	1	6	3	
	3	5	4	3	3	1	5	3	
	4	12	3	2	3	1	3	3	
	5	25	3	2	2	1	3	3	
	6	29	3	2	2	1	2	3	
25	1	4	4	2	1	4	6	18	
	2	5	4	2	1	4	5	18	
	3	6	4	2	1	4	5	17	
	4	23	3	2	1	3	5	17	
	5	28	2	1	1	2	4	17	
	6	29	2	1	1	2	4	16	
26	1	8	5	5	5	2	22	29	
	2	18	3	4	5	2	21	24	
	3	21	3	4	5	2	21	21	
	4	23	2	3	5	2	20	16	
	5	26	1	3	5	2	20	11	
	6	27	1	3	5	2	20	6	
27	1	2	5	5	4	4	25	16	
	2	8	5	4	3	3	23	14	
	3	9	5	3	3	3	21	12	
	4	11	5	3	2	3	18	12	
	5	12	5	2	1	3	15	10	
	6	15	5	2	1	3	14	9	
28	1	2	2	2	3	2	23	28	
	2	5	1	2	3	2	19	26	
	3	8	1	2	2	2	15	24	
	4	13	1	2	2	2	15	22	
	5	14	1	1	1	2	12	20	
	6	29	1	1	1	2	7	17	
29	1	2	5	4	4	4	29	15	
	2	3	4	4	4	4	26	15	
	3	8	3	4	3	4	26	15	
	4	13	3	4	3	3	23	15	
	5	14	2	4	2	3	22	15	
	6	18	2	4	2	3	21	15	
30	1	7	4	4	3	3	20	23	
	2	16	3	4	2	3	19	20	
	3	19	3	4	2	3	18	16	
	4	21	2	3	2	3	18	14	
	5	22	2	2	2	3	17	12	
	6	23	1	2	2	3	16	11	
31	1	13	2	5	1	2	19	25	
	2	14	1	4	1	2	16	22	
	3	20	1	4	1	2	16	21	
	4	22	1	3	1	2	13	16	
	5	25	1	3	1	2	12	13	
	6	29	1	2	1	2	12	11	
32	1	1	4	4	4	1	13	26	
	2	2	4	4	3	1	13	20	
	3	4	3	4	3	1	12	17	
	4	7	3	4	2	1	12	14	
	5	23	2	4	2	1	11	12	
	6	26	2	4	2	1	10	7	
33	1	4	1	5	1	5	27	28	
	2	5	1	5	1	3	25	26	
	3	6	1	5	1	3	22	25	
	4	10	1	5	1	3	16	24	
	5	21	1	5	1	1	13	22	
	6	22	1	5	1	1	11	20	
34	1	4	5	3	4	1	13	29	
	2	5	4	3	3	1	11	26	
	3	9	4	3	3	1	10	23	
	4	10	4	3	3	1	9	21	
	5	14	3	2	2	1	7	21	
	6	15	3	2	2	1	4	18	
35	1	5	5	2	3	2	11	20	
	2	12	4	1	3	2	10	19	
	3	15	3	1	2	2	9	14	
	4	20	3	1	2	2	8	10	
	5	27	1	1	1	2	7	8	
	6	29	1	1	1	2	7	4	
36	1	9	2	4	4	2	30	25	
	2	11	2	4	3	2	24	22	
	3	15	2	4	3	2	21	18	
	4	17	2	4	3	2	14	16	
	5	23	1	4	2	2	13	10	
	6	30	1	4	2	2	8	6	
37	1	7	3	4	4	3	16	14	
	2	10	3	3	3	3	15	13	
	3	12	3	3	3	3	15	12	
	4	14	3	2	2	3	13	12	
	5	16	3	2	2	2	12	12	
	6	19	3	2	2	2	11	11	
38	1	14	2	3	2	2	15	8	
	2	15	2	3	1	2	15	7	
	3	15	2	2	1	2	14	8	
	4	16	2	2	1	2	14	7	
	5	17	1	1	1	2	13	6	
	6	26	1	1	1	2	13	5	
39	1	1	4	3	4	2	21	28	
	2	20	4	2	3	2	17	27	
	3	21	4	2	3	2	13	27	
	4	22	3	2	2	2	12	27	
	5	25	3	2	2	2	6	27	
	6	29	3	2	1	2	6	27	
40	1	5	1	5	2	3	13	29	
	2	13	1	4	2	3	13	29	
	3	23	1	4	2	3	13	28	
	4	25	1	4	2	2	12	28	
	5	28	1	3	2	1	11	27	
	6	29	1	3	2	1	11	26	
41	1	1	5	5	2	4	26	10	
	2	3	4	4	2	4	26	9	
	3	9	4	4	2	3	25	8	
	4	12	3	3	2	3	23	8	
	5	15	3	3	2	2	23	6	
	6	20	2	3	2	1	22	6	
42	1	3	3	3	2	4	15	29	
	2	6	3	3	1	3	15	24	
	3	11	3	3	1	3	12	23	
	4	14	3	3	1	3	9	19	
	5	15	3	3	1	3	9	17	
	6	16	3	3	1	3	5	15	
43	1	1	4	4	1	2	29	8	
	2	3	4	3	1	2	29	7	
	3	10	3	3	1	2	28	6	
	4	11	3	2	1	2	27	5	
	5	24	2	2	1	2	26	4	
	6	30	1	2	1	2	26	3	
44	1	5	1	4	4	3	24	1	
	2	8	1	4	4	3	19	1	
	3	15	1	4	4	3	15	1	
	4	18	1	3	4	3	14	1	
	5	19	1	3	4	3	7	1	
	6	22	1	2	4	3	3	1	
45	1	4	4	5	4	4	25	18	
	2	6	4	4	4	3	23	17	
	3	21	3	4	4	3	21	17	
	4	26	3	4	4	2	16	16	
	5	28	2	3	4	2	16	13	
	6	29	1	3	4	2	11	12	
46	1	1	2	3	3	2	23	8	
	2	9	2	3	2	2	21	7	
	3	12	2	3	2	2	18	7	
	4	14	2	3	2	2	15	6	
	5	18	2	3	2	1	15	5	
	6	30	2	3	2	1	13	3	
47	1	4	5	4	4	4	25	11	
	2	6	5	4	3	4	21	8	
	3	8	5	4	3	3	18	7	
	4	9	5	3	3	3	11	5	
	5	24	5	2	3	1	8	4	
	6	26	5	2	3	1	5	3	
48	1	5	2	3	2	3	21	20	
	2	9	2	2	2	3	19	19	
	3	14	2	2	2	3	16	15	
	4	21	1	2	2	3	14	14	
	5	28	1	2	2	3	14	12	
	6	30	1	2	2	3	12	8	
49	1	6	4	1	5	3	25	28	
	2	7	3	1	5	2	21	28	
	3	17	3	1	5	2	19	28	
	4	20	3	1	5	2	18	27	
	5	21	3	1	5	2	13	27	
	6	22	3	1	5	2	13	26	
50	1	4	5	4	5	2	7	22	
	2	5	4	4	3	1	6	21	
	3	26	4	4	3	1	6	19	
	4	27	3	3	2	1	6	18	
	5	28	3	3	1	1	5	15	
	6	30	3	3	1	1	5	14	
51	1	1	5	4	2	4	19	23	
	2	21	4	4	1	4	17	22	
	3	26	4	3	1	3	16	19	
	4	27	4	2	1	3	15	13	
	5	28	4	2	1	1	11	10	
	6	29	4	1	1	1	9	10	
52	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	R 3	R 4	N 1	N 2
	36	33	29	32	816	789

************************************************************************
