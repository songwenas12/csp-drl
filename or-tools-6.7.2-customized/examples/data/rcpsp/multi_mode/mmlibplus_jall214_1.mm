jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 4 R
- nonrenewable              : 4 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	7		2 3 4 5 6 8 9 
2	6	6		16 15 14 13 12 7 
3	6	5		21 16 14 12 7 
4	6	5		21 16 15 12 7 
5	6	4		21 14 12 7 
6	6	6		23 16 15 14 13 12 
7	6	3		23 11 10 
8	6	3		23 18 11 
9	6	3		23 16 14 
10	6	3		26 19 18 
11	6	2		19 17 
12	6	2		20 17 
13	6	2		20 18 
14	6	3		26 24 19 
15	6	3		26 24 19 
16	6	1		17 
17	6	5		29 27 26 24 22 
18	6	5		31 27 25 24 22 
19	6	5		33 31 27 25 22 
20	6	4		27 26 25 22 
21	6	6		41 36 34 33 30 29 
22	6	5		41 36 34 32 28 
23	6	5		41 35 34 31 28 
24	6	4		41 34 33 28 
25	6	3		41 30 29 
26	6	4		37 35 34 31 
27	6	4		43 38 37 35 
28	6	2		43 30 
29	6	3		43 38 35 
30	6	3		39 38 37 
31	6	3		43 39 38 
32	6	3		43 39 38 
33	6	2		43 35 
34	6	3		51 43 38 
35	6	3		42 40 39 
36	6	3		42 40 39 
37	6	5		51 50 49 47 42 
38	6	3		49 42 40 
39	6	5		51 50 49 47 44 
40	6	4		50 48 47 44 
41	6	4		49 48 47 44 
42	6	3		48 45 44 
43	6	3		48 47 46 
44	6	1		46 
45	6	1		46 
46	6	1		52 
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
2	1	2	4	4	2	1	26	26	21	27	
	2	4	4	4	2	1	24	24	18	27	
	3	6	4	3	2	1	24	23	17	25	
	4	16	3	2	1	1	23	21	13	25	
	5	18	3	2	1	1	22	21	11	24	
	6	23	3	1	1	1	21	19	9	22	
3	1	2	5	5	4	2	18	25	26	23	
	2	3	4	4	3	1	17	23	26	17	
	3	18	4	4	3	1	16	23	23	13	
	4	21	4	4	2	1	15	21	22	10	
	5	26	3	4	1	1	12	20	21	6	
	6	28	3	4	1	1	11	18	20	6	
4	1	2	5	2	3	2	18	26	17	29	
	2	4	4	2	3	1	18	20	16	24	
	3	7	4	2	3	1	18	14	16	24	
	4	8	3	2	2	1	18	12	16	20	
	5	24	3	1	2	1	18	8	15	19	
	6	25	3	1	2	1	18	6	15	18	
5	1	4	4	4	3	5	12	27	16	16	
	2	9	4	4	3	3	11	26	16	15	
	3	12	4	4	3	3	11	26	15	13	
	4	14	3	4	3	3	9	23	15	11	
	5	18	2	4	3	2	7	23	15	8	
	6	30	2	4	3	1	6	22	14	7	
6	1	2	5	3	3	2	26	20	25	22	
	2	9	4	3	2	2	22	20	22	19	
	3	14	4	3	2	2	19	20	20	16	
	4	25	4	3	2	2	19	20	20	11	
	5	28	4	3	2	2	16	20	16	10	
	6	29	4	3	2	2	13	20	15	7	
7	1	10	4	4	2	2	28	8	21	5	
	2	11	4	4	2	2	21	6	21	4	
	3	12	3	4	2	2	16	6	21	4	
	4	13	3	4	2	2	13	5	20	3	
	5	14	3	4	1	2	12	4	20	3	
	6	29	2	4	1	2	7	3	20	3	
8	1	5	3	3	2	2	26	26	27	24	
	2	12	3	3	1	1	25	26	24	22	
	3	14	2	3	1	1	23	25	21	20	
	4	16	2	2	1	1	21	22	19	16	
	5	20	1	2	1	1	21	22	15	11	
	6	27	1	2	1	1	19	20	10	10	
9	1	1	3	4	4	5	16	19	9	25	
	2	4	3	4	4	4	14	19	8	18	
	3	6	3	4	3	4	14	19	7	15	
	4	21	3	3	3	3	12	19	6	12	
	5	23	3	2	2	2	12	19	5	12	
	6	27	3	2	2	2	11	19	3	8	
10	1	11	4	2	4	4	26	15	29	13	
	2	16	4	2	4	4	26	11	25	11	
	3	20	4	2	4	4	26	11	22	8	
	4	26	3	2	4	3	26	10	18	6	
	5	27	3	2	3	3	26	8	13	4	
	6	28	3	2	3	3	26	6	8	3	
11	1	2	5	5	1	3	28	21	16	15	
	2	9	5	4	1	3	25	20	15	15	
	3	13	5	4	1	3	19	14	14	15	
	4	14	5	3	1	2	19	14	13	15	
	5	16	5	2	1	2	14	10	11	15	
	6	26	5	2	1	2	11	4	11	15	
12	1	9	4	3	4	5	18	25	28	22	
	2	10	3	3	3	3	15	24	22	21	
	3	12	3	3	3	3	14	17	18	21	
	4	17	2	3	2	2	12	15	15	21	
	5	18	1	3	1	1	12	13	10	21	
	6	22	1	3	1	1	11	9	6	21	
13	1	7	2	3	5	2	8	13	28	25	
	2	11	2	2	4	2	8	12	24	24	
	3	15	2	2	4	2	7	9	22	24	
	4	16	2	2	4	1	6	9	16	22	
	5	17	2	2	3	1	6	5	14	21	
	6	23	2	2	3	1	5	4	13	21	
14	1	3	2	4	2	4	20	23	13	18	
	2	12	2	3	2	3	18	21	11	14	
	3	14	2	3	2	3	18	19	9	13	
	4	17	2	2	2	2	15	17	7	10	
	5	24	2	2	2	2	12	15	4	9	
	6	28	2	2	2	2	10	14	3	5	
15	1	2	4	4	4	3	12	14	16	21	
	2	4	4	4	4	3	11	11	12	20	
	3	5	4	4	4	2	10	10	9	17	
	4	13	3	4	4	2	9	6	8	15	
	5	21	2	4	4	1	9	4	5	15	
	6	28	2	4	4	1	8	4	2	12	
16	1	1	1	3	4	1	17	12	25	28	
	2	12	1	2	4	1	14	12	24	22	
	3	17	1	2	4	1	13	12	24	20	
	4	18	1	2	3	1	10	11	24	16	
	5	23	1	2	3	1	9	11	24	14	
	6	25	1	2	2	1	5	11	24	9	
17	1	1	5	4	5	5	20	5	27	17	
	2	4	3	3	4	4	20	5	24	17	
	3	15	3	3	4	3	18	4	20	16	
	4	22	2	2	4	3	16	4	19	13	
	5	29	2	2	3	2	16	4	16	12	
	6	30	1	2	3	2	14	3	12	11	
18	1	1	3	1	4	5	19	13	12	12	
	2	6	3	1	4	5	18	11	11	9	
	3	12	3	1	4	5	16	10	10	9	
	4	16	3	1	4	5	13	9	10	6	
	5	17	2	1	3	5	6	7	8	6	
	6	19	2	1	3	5	6	5	8	3	
19	1	2	4	4	2	5	13	27	19	29	
	2	5	3	4	1	5	13	27	18	28	
	3	11	3	4	1	5	12	24	18	27	
	4	19	2	4	1	5	11	23	17	26	
	5	27	2	4	1	5	10	20	15	25	
	6	28	2	4	1	5	8	18	14	25	
20	1	5	4	4	5	2	13	20	25	16	
	2	11	4	3	4	2	12	20	22	16	
	3	14	4	3	4	2	11	19	21	14	
	4	17	4	2	3	2	10	19	18	13	
	5	25	3	1	3	2	10	18	16	12	
	6	28	3	1	3	2	9	18	14	12	
21	1	4	5	1	4	2	28	29	17	19	
	2	5	4	1	3	2	26	29	17	18	
	3	10	4	1	3	2	25	29	16	17	
	4	23	4	1	3	2	22	28	13	17	
	5	24	4	1	3	2	20	28	13	17	
	6	25	4	1	3	2	17	28	11	16	
22	1	2	5	3	3	2	17	4	7	18	
	2	5	4	3	2	2	12	4	7	17	
	3	21	4	3	2	2	11	4	7	15	
	4	22	3	3	1	2	9	4	7	12	
	5	23	3	3	1	2	7	4	6	6	
	6	25	3	3	1	2	6	4	6	6	
23	1	5	4	4	4	1	19	14	29	20	
	2	12	4	3	4	1	19	14	25	18	
	3	14	4	3	4	1	18	14	25	15	
	4	18	3	2	4	1	17	14	21	15	
	5	22	3	1	4	1	15	14	20	13	
	6	30	3	1	4	1	15	14	19	9	
24	1	7	5	1	4	4	19	6	7	14	
	2	8	4	1	4	4	17	5	6	14	
	3	12	4	1	3	4	16	5	6	14	
	4	15	4	1	3	4	12	5	5	14	
	5	16	3	1	1	4	11	3	5	14	
	6	23	3	1	1	4	10	3	5	14	
25	1	2	5	5	3	2	23	17	25	16	
	2	19	4	4	2	2	23	14	21	16	
	3	20	4	4	2	2	22	14	18	15	
	4	21	3	4	2	2	20	13	13	15	
	5	27	3	4	2	2	19	12	8	13	
	6	29	3	4	2	2	19	11	5	13	
26	1	3	2	5	4	4	27	22	20	23	
	2	4	2	4	3	3	22	22	17	21	
	3	6	2	4	3	3	22	22	15	18	
	4	11	2	4	3	2	14	22	13	17	
	5	21	2	4	2	2	13	22	9	12	
	6	25	2	4	2	1	9	22	4	11	
27	1	2	4	5	4	3	20	13	22	22	
	2	4	4	3	4	3	19	12	21	20	
	3	6	4	3	4	3	16	10	18	17	
	4	16	4	2	3	2	14	8	16	17	
	5	23	4	2	3	2	12	8	15	14	
	6	29	4	1	2	2	12	7	12	14	
28	1	1	4	2	3	5	26	29	23	27	
	2	5	4	2	3	4	23	21	20	25	
	3	8	4	2	2	4	22	16	17	25	
	4	12	3	2	2	3	20	16	14	22	
	5	18	2	2	2	3	17	10	12	20	
	6	26	2	2	1	3	15	3	7	17	
29	1	7	1	4	4	5	23	17	5	13	
	2	17	1	4	4	3	18	16	4	12	
	3	18	1	4	4	3	15	16	3	8	
	4	19	1	4	3	3	11	15	3	7	
	5	24	1	3	2	1	8	15	2	4	
	6	25	1	3	2	1	4	15	1	2	
30	1	2	3	4	4	2	11	15	11	18	
	2	6	3	4	3	2	11	14	10	17	
	3	14	3	4	3	2	11	12	8	16	
	4	20	3	3	3	2	11	9	7	16	
	5	21	3	3	3	2	11	8	6	15	
	6	24	3	2	3	2	11	6	5	15	
31	1	7	4	4	2	4	18	13	19	8	
	2	12	4	4	2	3	16	13	16	8	
	3	13	4	3	2	3	14	10	16	7	
	4	15	3	3	2	3	12	6	15	4	
	5	20	3	2	2	2	11	3	13	2	
	6	30	2	2	2	2	9	2	12	1	
32	1	6	2	5	4	5	15	20	26	19	
	2	9	2	4	4	4	14	17	23	19	
	3	13	2	3	4	4	14	13	21	19	
	4	22	2	3	3	4	13	11	18	19	
	5	23	2	2	3	4	11	9	15	19	
	6	26	2	2	2	4	10	7	10	19	
33	1	10	3	3	4	5	12	25	22	23	
	2	15	3	3	4	4	11	19	21	21	
	3	16	3	3	4	4	8	15	17	20	
	4	18	3	3	3	4	7	13	14	16	
	5	26	2	2	2	3	5	11	13	14	
	6	30	2	2	2	3	1	8	10	11	
34	1	4	5	3	4	1	20	18	17	20	
	2	5	4	2	3	1	20	17	15	19	
	3	12	4	2	3	1	20	17	13	18	
	4	21	3	2	3	1	20	16	13	16	
	5	22	3	2	3	1	20	16	10	15	
	6	24	3	2	3	1	20	16	8	15	
35	1	5	5	3	4	4	26	23	23	27	
	2	6	3	2	3	4	26	22	20	21	
	3	8	3	2	3	4	25	19	19	16	
	4	16	2	2	2	4	23	18	11	10	
	5	17	1	1	2	3	21	16	10	9	
	6	20	1	1	2	3	21	13	3	4	
36	1	2	4	5	4	2	17	29	8	25	
	2	7	4	4	3	2	16	21	6	22	
	3	12	4	3	2	2	15	16	5	21	
	4	23	4	3	2	2	14	11	4	18	
	5	24	3	2	1	1	13	6	3	14	
	6	25	3	2	1	1	13	2	1	11	
37	1	2	3	4	2	4	9	21	15	20	
	2	3	3	4	2	3	7	18	13	19	
	3	6	3	4	2	3	5	18	12	17	
	4	15	2	3	2	3	4	15	12	16	
	5	19	1	3	2	3	3	15	11	16	
	6	24	1	3	2	3	2	13	9	15	
38	1	1	4	3	5	5	25	27	20	19	
	2	2	4	3	4	5	20	26	17	17	
	3	6	3	3	3	5	16	23	14	15	
	4	18	3	3	2	5	12	20	9	11	
	5	22	3	3	2	5	7	20	4	10	
	6	29	2	3	1	5	4	17	4	9	
39	1	7	1	3	3	5	28	16	12	14	
	2	9	1	3	2	4	23	13	11	13	
	3	11	1	3	2	4	17	10	10	12	
	4	17	1	2	2	4	15	7	9	12	
	5	25	1	1	2	4	8	7	9	12	
	6	29	1	1	2	4	5	4	8	11	
40	1	3	3	3	5	2	11	25	26	10	
	2	12	3	3	3	2	11	24	25	9	
	3	14	3	3	3	2	11	19	20	7	
	4	15	3	3	2	2	11	13	20	5	
	5	21	3	2	1	2	10	11	14	3	
	6	29	3	2	1	2	10	6	12	1	
41	1	2	5	3	4	1	22	28	11	24	
	2	4	4	2	4	1	20	26	11	19	
	3	8	3	2	4	1	20	25	10	17	
	4	11	3	2	4	1	18	23	8	10	
	5	12	3	2	4	1	16	20	7	7	
	6	18	2	2	4	1	14	20	5	4	
42	1	6	5	2	2	4	24	15	28	22	
	2	19	5	2	2	3	22	14	20	19	
	3	23	5	2	2	3	20	11	18	15	
	4	24	5	2	1	3	15	11	12	14	
	5	25	5	1	1	3	14	8	8	11	
	6	27	5	1	1	3	8	7	6	9	
43	1	3	2	3	5	5	24	28	24	24	
	2	7	2	3	4	4	24	23	23	22	
	3	8	2	3	4	4	21	21	22	22	
	4	21	1	2	3	4	18	19	21	20	
	5	26	1	2	3	3	16	16	20	16	
	6	30	1	1	3	3	14	15	20	14	
44	1	3	5	5	4	1	18	28	16	14	
	2	4	4	4	4	1	17	25	15	11	
	3	5	4	4	4	1	14	20	13	11	
	4	6	3	4	4	1	13	17	12	6	
	5	7	2	4	4	1	10	16	12	5	
	6	23	2	4	4	1	10	11	11	2	
45	1	4	4	5	2	1	25	29	20	24	
	2	11	4	5	1	1	21	24	19	21	
	3	15	4	5	1	1	16	21	19	16	
	4	16	4	5	1	1	12	17	18	13	
	5	24	4	5	1	1	12	11	18	9	
	6	25	4	5	1	1	7	9	17	8	
46	1	4	4	3	4	3	21	13	28	9	
	2	7	4	2	3	3	20	11	27	8	
	3	10	3	2	2	3	20	11	27	8	
	4	19	2	2	2	3	20	11	26	7	
	5	22	2	2	2	3	19	10	26	6	
	6	27	1	2	1	3	19	9	26	6	
47	1	5	5	3	2	3	9	19	8	9	
	2	6	4	2	2	3	8	18	7	7	
	3	7	4	2	2	3	8	17	7	6	
	4	16	4	2	1	3	8	17	7	5	
	5	17	3	1	1	3	8	16	6	4	
	6	23	3	1	1	3	8	16	5	4	
48	1	10	1	4	4	2	24	15	21	8	
	2	15	1	4	3	2	21	15	16	8	
	3	16	1	3	3	2	18	13	13	8	
	4	17	1	2	2	2	18	13	11	8	
	5	18	1	2	2	2	15	10	7	8	
	6	27	1	1	2	2	14	10	4	8	
49	1	2	4	4	4	1	22	28	27	13	
	2	3	4	4	3	1	18	26	26	12	
	3	9	4	4	3	1	16	26	25	11	
	4	10	4	4	3	1	15	26	25	10	
	5	17	3	4	3	1	9	24	25	7	
	6	23	3	4	3	1	9	24	24	6	
50	1	6	3	5	5	1	29	29	26	18	
	2	7	2	4	3	1	26	28	21	15	
	3	13	2	4	3	1	26	26	18	14	
	4	14	2	4	2	1	24	25	18	13	
	5	22	1	3	2	1	21	24	15	11	
	6	30	1	3	1	1	19	23	12	11	
51	1	6	2	4	5	5	23	6	25	5	
	2	13	2	4	3	4	23	6	22	5	
	3	22	2	3	3	4	21	6	17	4	
	4	23	2	2	2	4	20	6	13	4	
	5	24	2	1	2	4	19	5	8	3	
	6	28	2	1	1	4	19	5	4	3	
52	1	0	0	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	R 3	R 4	N 1	N 2	N 3	N 4
	23	23	20	20	679	676	627	613

************************************************************************
