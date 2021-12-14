jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 4 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	8		2 3 4 5 7 9 10 11 
2	3	9		21 20 19 16 15 14 13 8 6 
3	3	8		31 23 20 19 17 16 15 8 
4	3	8		33 23 22 20 19 18 16 12 
5	3	10		35 31 28 27 25 22 21 19 17 15 
6	3	9		39 35 31 29 27 24 23 18 17 
7	3	6		33 25 24 22 21 18 
8	3	3		22 18 12 
9	3	8		51 35 33 27 25 23 22 20 
10	3	7		51 35 25 24 23 22 20 
11	3	12		51 50 49 39 38 36 35 34 33 30 29 26 
12	3	9		51 50 36 35 34 30 29 28 25 
13	3	4		33 29 23 18 
14	3	7		50 39 31 30 29 27 24 
15	3	8		51 49 38 36 33 30 29 26 
16	3	6		51 50 34 29 27 26 
17	3	6		51 50 48 36 33 30 
18	3	5		51 50 38 34 26 
19	3	11		51 49 48 47 46 45 43 39 38 37 34 
20	3	6		50 49 48 47 34 30 
21	3	6		50 49 47 38 36 30 
22	3	4		39 38 30 29 
23	3	4		49 48 47 30 
24	3	6		48 47 44 42 36 32 
25	3	9		49 47 46 45 44 43 42 38 37 
26	3	5		48 47 44 42 32 
27	3	5		49 47 44 42 32 
28	3	10		49 48 47 46 45 44 43 42 41 40 
29	3	4		47 44 42 32 
30	3	3		44 42 32 
31	3	6		48 46 44 42 41 37 
32	3	5		46 45 43 41 37 
33	3	4		47 43 41 37 
34	3	4		44 42 41 40 
35	3	3		45 43 37 
36	3	1		37 
37	3	1		40 
38	3	1		41 
39	3	1		42 
40	3	1		52 
41	3	1		52 
42	3	1		52 
43	3	1		52 
44	3	1		52 
45	3	1		52 
46	3	1		52 
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
2	1	7	3	4	2	5	28	18	
	2	15	3	4	1	3	25	17	
	3	20	1	4	1	3	23	14	
3	1	6	3	2	3	4	17	18	
	2	11	3	2	2	3	17	13	
	3	28	3	2	2	2	13	11	
4	1	6	3	4	2	3	29	25	
	2	18	3	2	2	2	27	21	
	3	21	3	2	2	2	23	11	
5	1	5	2	4	5	4	15	9	
	2	13	1	4	4	3	13	7	
	3	16	1	4	4	2	9	5	
6	1	1	2	5	4	3	23	26	
	2	10	2	4	2	2	20	17	
	3	11	2	3	2	2	18	16	
7	1	8	4	1	3	4	24	12	
	2	11	4	1	2	4	24	8	
	3	12	4	1	2	3	23	5	
8	1	2	4	4	4	5	13	2	
	2	12	3	3	4	3	11	2	
	3	26	2	2	2	3	8	1	
9	1	9	5	3	5	4	19	25	
	2	17	5	1	4	3	14	23	
	3	30	5	1	3	3	12	19	
10	1	5	3	3	3	1	3	6	
	2	6	2	3	2	1	2	5	
	3	30	1	3	1	1	2	3	
11	1	10	1	5	2	2	5	4	
	2	21	1	5	2	1	3	3	
	3	23	1	5	2	1	2	3	
12	1	5	2	3	5	5	9	24	
	2	9	2	1	3	4	7	24	
	3	29	1	1	3	3	4	23	
13	1	3	1	1	2	4	8	20	
	2	20	1	1	2	2	7	14	
	3	28	1	1	1	2	3	6	
14	1	5	3	4	4	1	18	10	
	2	6	3	3	3	1	15	9	
	3	16	2	3	3	1	10	9	
15	1	11	2	1	1	2	9	25	
	2	13	2	1	1	2	6	24	
	3	16	2	1	1	1	3	24	
16	1	7	5	1	5	4	22	25	
	2	15	3	1	4	4	15	25	
	3	16	3	1	4	3	3	25	
17	1	19	5	3	2	3	16	18	
	2	20	3	3	2	2	14	17	
	3	23	3	3	2	2	11	17	
18	1	2	3	5	5	4	13	23	
	2	28	3	5	5	3	12	17	
	3	30	2	5	5	2	11	6	
19	1	6	4	4	5	3	12	15	
	2	12	3	3	3	3	12	14	
	3	24	2	3	3	2	10	11	
20	1	9	3	2	5	5	18	24	
	2	13	3	2	3	3	18	24	
	3	29	2	1	3	3	15	24	
21	1	25	3	3	2	4	24	3	
	2	29	2	3	2	4	16	3	
	3	30	1	3	2	2	9	2	
22	1	8	4	4	4	3	13	29	
	2	15	4	4	4	2	12	19	
	3	18	4	4	4	1	4	16	
23	1	6	5	4	4	1	22	8	
	2	11	5	4	2	1	22	8	
	3	19	5	4	2	1	20	5	
24	1	3	4	2	5	2	29	22	
	2	14	2	1	3	1	27	10	
	3	29	2	1	2	1	23	5	
25	1	2	5	2	4	3	17	21	
	2	21	4	2	4	3	16	17	
	3	27	3	2	3	3	16	14	
26	1	3	4	3	3	1	19	19	
	2	4	3	3	1	1	11	15	
	3	18	2	2	1	1	9	13	
27	1	1	1	4	4	3	20	11	
	2	10	1	3	3	3	17	7	
	3	18	1	2	3	3	16	6	
28	1	6	4	3	4	4	29	12	
	2	13	3	3	4	3	25	11	
	3	14	3	2	2	2	24	10	
29	1	4	4	4	3	4	13	20	
	2	6	4	4	2	3	13	19	
	3	26	3	4	2	1	13	18	
30	1	10	5	5	4	2	16	29	
	2	17	5	4	3	1	16	26	
	3	21	5	4	3	1	15	25	
31	1	6	3	5	3	4	17	12	
	2	10	3	3	2	2	14	5	
	3	29	1	3	2	2	10	4	
32	1	2	3	3	4	5	22	23	
	2	23	2	1	3	5	14	22	
	3	24	2	1	2	5	8	20	
33	1	1	3	2	5	2	20	17	
	2	17	2	1	4	1	19	16	
	3	28	2	1	3	1	15	16	
34	1	3	2	2	2	4	14	9	
	2	9	2	1	1	4	9	9	
	3	20	2	1	1	4	7	8	
35	1	7	3	4	4	2	20	12	
	2	8	2	4	4	2	14	11	
	3	27	2	4	4	1	4	8	
36	1	14	3	2	5	4	11	27	
	2	17	3	2	4	2	10	18	
	3	18	2	1	4	2	7	13	
37	1	6	4	2	5	2	7	20	
	2	27	4	1	4	1	5	14	
	3	28	4	1	4	1	5	13	
38	1	7	1	2	3	2	2	26	
	2	8	1	1	2	2	1	22	
	3	21	1	1	1	1	1	22	
39	1	5	5	4	3	4	24	26	
	2	17	5	3	1	3	14	25	
	3	22	5	3	1	1	8	25	
40	1	10	3	5	3	2	21	26	
	2	20	2	4	3	2	15	25	
	3	25	1	4	3	2	7	25	
41	1	9	5	2	4	4	17	13	
	2	14	4	2	3	3	16	14	
	3	15	4	2	3	3	16	13	
42	1	2	3	5	2	4	19	17	
	2	28	2	4	2	3	18	14	
	3	29	2	2	2	2	16	12	
43	1	11	5	1	5	3	16	13	
	2	13	4	1	2	3	15	6	
	3	28	4	1	1	2	14	5	
44	1	9	3	5	4	5	19	20	
	2	15	3	4	2	4	16	17	
	3	20	3	4	2	4	16	13	
45	1	20	3	1	3	5	12	10	
	2	21	3	1	1	5	10	8	
	3	26	1	1	1	5	7	7	
46	1	19	1	2	4	3	20	18	
	2	20	1	2	4	2	20	17	
	3	26	1	1	4	2	19	2	
47	1	18	5	3	4	3	12	30	
	2	19	4	3	3	3	11	28	
	3	22	3	3	3	3	10	24	
48	1	3	3	4	3	4	25	17	
	2	6	2	4	3	3	17	15	
	3	9	2	4	2	3	9	13	
49	1	9	5	5	2	5	13	20	
	2	10	4	3	2	3	12	17	
	3	15	4	2	2	2	12	15	
50	1	1	5	3	3	5	21	20	
	2	4	4	1	3	4	20	16	
	3	19	4	1	3	2	18	6	
51	1	9	5	3	5	3	19	18	
	2	13	5	3	4	2	17	14	
	3	19	5	3	3	1	11	12	
52	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	R 3	R 4	N 1	N 2
	23	21	23	20	783	828

************************************************************************
