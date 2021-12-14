jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 4 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	7		2 3 4 8 10 13 24 
2	6	7		26 23 20 16 12 6 5 
3	6	10		51 30 25 21 20 19 18 15 14 9 
4	6	8		37 27 25 21 20 19 17 11 
5	6	4		51 37 15 7 
6	6	7		30 25 22 21 19 17 14 
7	6	9		35 31 30 27 25 22 21 19 18 
8	6	9		35 30 28 27 25 23 22 21 18 
9	6	6		35 29 27 26 22 16 
10	6	8		41 37 34 31 30 29 28 22 
11	6	4		35 26 18 16 
12	6	4		31 29 22 21 
13	6	8		50 41 40 39 35 34 31 28 
14	6	7		50 41 39 37 35 34 28 
15	6	7		50 41 39 36 34 33 29 
16	6	6		50 41 36 34 31 28 
17	6	7		51 41 36 34 33 32 31 
18	6	6		50 41 36 34 33 29 
19	6	5		50 41 39 29 28 
20	6	8		50 48 46 41 40 38 33 32 
21	6	7		50 46 41 38 34 33 32 
22	6	6		50 48 39 36 33 32 
23	6	5		49 36 34 32 31 
24	6	5		50 49 40 31 29 
25	6	5		49 44 41 39 34 
26	6	5		49 46 40 38 34 
27	6	5		50 46 40 38 32 
28	6	4		46 38 33 32 
29	6	4		48 46 38 32 
30	6	5		50 48 46 45 38 
31	6	4		48 46 45 38 
32	6	5		47 45 44 43 42 
33	6	4		49 47 43 42 
34	6	4		48 47 45 42 
35	6	4		48 47 46 45 
36	6	3		44 42 40 
37	6	3		44 42 40 
38	6	3		44 43 42 
39	6	2		46 43 
40	6	1		43 
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
jobnr.	mode	dur	R1	R2	N1	N2	N3	N4	
------------------------------------------------------------------------
1	1	0	0	0	0	0	0	0	
2	1	5	2	2	19	12	17	11	
	2	6	2	2	18	11	14	8	
	3	9	2	2	17	11	10	8	
	4	10	2	2	17	10	8	5	
	5	15	2	2	16	9	5	3	
	6	16	2	2	16	9	1	2	
3	1	6	5	2	16	19	12	15	
	2	7	5	1	14	16	10	11	
	3	9	5	1	12	14	8	11	
	4	12	5	1	9	11	7	10	
	5	19	5	1	7	8	6	8	
	6	20	5	1	7	5	5	5	
4	1	2	3	5	13	7	18	9	
	2	3	3	5	10	7	15	7	
	3	4	3	5	10	6	14	7	
	4	12	3	5	8	6	11	6	
	5	14	2	5	4	5	10	4	
	6	15	2	5	4	5	9	3	
5	1	1	5	2	5	13	17	14	
	2	2	4	2	4	10	13	13	
	3	8	4	2	4	9	12	12	
	4	13	4	2	4	9	10	11	
	5	16	3	2	3	6	6	10	
	6	20	3	2	3	5	3	10	
6	1	4	2	2	11	9	18	11	
	2	7	2	2	10	7	15	10	
	3	8	2	2	9	6	15	9	
	4	11	1	1	7	6	13	9	
	5	12	1	1	6	4	12	7	
	6	16	1	1	5	3	11	5	
7	1	4	4	4	3	7	16	18	
	2	7	4	3	3	7	16	17	
	3	12	4	3	3	6	16	16	
	4	14	4	3	3	4	16	16	
	5	15	3	3	3	3	15	16	
	6	20	3	3	3	3	15	15	
8	1	3	3	4	8	5	13	17	
	2	7	3	3	7	4	12	16	
	3	8	3	3	6	3	11	15	
	4	9	2	3	6	3	10	15	
	5	16	2	3	4	2	9	14	
	6	18	1	3	4	2	9	12	
9	1	1	3	2	14	8	12	12	
	2	12	3	2	13	8	12	10	
	3	15	2	2	10	7	10	10	
	4	16	2	1	10	7	10	8	
	5	17	1	1	8	6	9	5	
	6	18	1	1	5	5	8	5	
10	1	2	5	5	19	4	18	10	
	2	5	4	4	18	4	15	8	
	3	6	4	4	18	4	11	7	
	4	7	4	4	17	4	10	7	
	5	8	3	4	17	4	8	5	
	6	11	3	4	16	4	3	3	
11	1	6	4	2	19	16	16	11	
	2	8	4	2	17	14	14	10	
	3	10	3	2	16	13	12	10	
	4	11	3	2	15	13	11	10	
	5	12	3	2	13	11	8	8	
	6	15	2	2	12	10	7	8	
12	1	4	5	4	16	8	17	9	
	2	5	4	3	16	7	15	9	
	3	10	4	3	14	7	13	8	
	4	11	3	3	13	7	12	7	
	5	19	3	2	12	7	10	5	
	6	20	3	2	11	7	7	5	
13	1	2	1	5	12	18	8	19	
	2	4	1	4	11	17	6	18	
	3	10	1	4	9	17	6	16	
	4	12	1	4	8	16	6	12	
	5	14	1	3	8	15	5	10	
	6	16	1	3	7	15	4	8	
14	1	2	4	4	17	5	13	5	
	2	3	4	4	15	5	12	4	
	3	5	4	4	15	5	10	4	
	4	7	4	3	12	4	7	4	
	5	8	4	2	10	4	6	3	
	6	18	4	2	8	4	3	3	
15	1	5	4	4	18	13	9	7	
	2	6	4	3	17	12	8	6	
	3	9	4	3	16	12	6	6	
	4	17	4	3	16	12	4	5	
	5	18	3	3	16	12	3	5	
	6	19	3	3	15	12	1	5	
16	1	6	3	5	4	12	17	17	
	2	7	3	5	4	12	17	16	
	3	8	3	5	4	12	17	13	
	4	10	3	5	4	12	17	12	
	5	14	3	5	4	11	17	11	
	6	18	3	5	4	11	17	9	
17	1	9	4	4	7	12	12	13	
	2	12	4	3	7	12	9	10	
	3	13	4	3	7	12	7	10	
	4	16	4	2	6	12	6	8	
	5	17	4	1	6	12	5	8	
	6	20	4	1	5	12	3	7	
18	1	3	4	1	12	4	13	7	
	2	4	4	1	12	4	12	6	
	3	6	4	1	12	4	10	6	
	4	7	4	1	11	3	9	5	
	5	9	4	1	10	3	7	5	
	6	13	4	1	10	3	5	4	
19	1	1	2	4	7	14	19	18	
	2	3	2	3	7	11	19	15	
	3	10	2	3	7	10	19	13	
	4	11	2	3	7	9	18	12	
	5	16	2	2	7	6	18	9	
	6	17	2	2	7	6	18	7	
20	1	5	4	1	20	17	20	11	
	2	6	3	1	18	14	18	11	
	3	8	3	1	15	11	16	10	
	4	9	3	1	14	7	16	10	
	5	16	3	1	12	4	14	8	
	6	18	3	1	12	4	13	8	
21	1	5	4	4	8	15	13	7	
	2	9	3	4	8	15	13	7	
	3	11	3	4	6	15	12	7	
	4	15	2	4	5	15	12	7	
	5	16	2	4	3	14	12	7	
	6	20	2	4	2	14	11	7	
22	1	4	4	4	15	12	11	16	
	2	7	4	4	15	10	11	15	
	3	9	4	4	15	7	9	12	
	4	12	4	4	15	5	9	7	
	5	14	4	4	14	4	8	6	
	6	16	4	4	14	2	6	3	
23	1	1	1	3	15	9	4	12	
	2	3	1	2	14	9	3	9	
	3	6	1	2	14	9	3	7	
	4	7	1	2	14	8	3	5	
	5	12	1	2	14	8	3	3	
	6	14	1	2	14	8	3	2	
24	1	2	3	4	18	8	18	18	
	2	7	2	4	16	7	16	17	
	3	9	2	3	16	7	16	15	
	4	11	2	3	13	7	15	15	
	5	12	2	3	11	5	13	14	
	6	18	2	2	10	5	13	13	
25	1	4	3	5	6	12	18	17	
	2	8	2	3	5	10	17	14	
	3	12	2	3	4	10	16	10	
	4	18	2	3	3	8	15	8	
	5	19	2	1	3	7	13	3	
	6	20	2	1	2	7	13	1	
26	1	4	3	5	14	10	19	7	
	2	5	2	4	12	8	18	7	
	3	7	2	3	12	7	18	7	
	4	10	2	3	9	5	18	7	
	5	15	1	3	7	5	18	6	
	6	20	1	2	6	3	18	6	
27	1	4	5	2	18	14	16	7	
	2	10	4	2	17	12	13	6	
	3	11	4	2	17	12	11	5	
	4	12	4	1	16	10	8	4	
	5	13	4	1	16	10	8	3	
	6	16	4	1	15	9	6	3	
28	1	1	3	4	18	6	4	5	
	2	3	3	4	18	5	3	4	
	3	4	3	4	17	5	3	4	
	4	7	3	4	15	4	3	4	
	5	15	3	4	14	2	3	4	
	6	16	3	4	13	1	3	4	
29	1	3	3	3	14	5	10	7	
	2	4	2	3	14	5	9	6	
	3	5	2	3	14	4	8	6	
	4	6	2	2	14	4	8	6	
	5	16	2	2	14	4	6	6	
	6	17	2	2	14	3	6	6	
30	1	3	3	2	16	18	17	16	
	2	5	2	2	14	18	14	15	
	3	6	2	2	14	17	13	14	
	4	7	2	1	13	17	11	12	
	5	12	2	1	11	15	8	10	
	6	17	2	1	10	15	6	10	
31	1	9	2	1	9	7	10	10	
	2	10	2	1	9	7	10	9	
	3	11	2	1	8	7	9	7	
	4	12	2	1	6	7	7	7	
	5	15	1	1	4	7	7	5	
	6	20	1	1	4	7	6	4	
32	1	1	4	5	13	8	15	12	
	2	5	4	4	12	7	15	12	
	3	7	3	3	11	7	15	11	
	4	9	2	3	8	6	15	9	
	5	15	1	2	8	6	15	8	
	6	18	1	2	5	6	15	7	
33	1	1	4	3	17	18	10	17	
	2	5	3	3	17	16	9	14	
	3	10	3	3	16	12	9	13	
	4	11	2	2	16	11	8	11	
	5	15	2	2	15	6	7	10	
	6	16	2	2	15	4	6	7	
34	1	7	5	3	18	16	9	17	
	2	11	5	2	18	14	8	15	
	3	12	5	2	17	11	8	13	
	4	14	5	1	15	11	8	13	
	5	15	5	1	13	5	8	11	
	6	18	5	1	11	4	8	10	
35	1	1	3	3	12	17	17	20	
	2	2	2	2	10	16	17	18	
	3	5	2	2	9	15	17	18	
	4	7	2	1	7	15	17	17	
	5	11	2	1	5	14	17	16	
	6	16	2	1	1	13	17	16	
36	1	11	5	4	10	1	8	6	
	2	13	4	3	10	1	7	5	
	3	14	4	3	10	1	5	4	
	4	15	4	2	9	1	5	4	
	5	16	4	2	9	1	4	3	
	6	17	4	1	8	1	3	3	
37	1	5	5	4	16	12	16	17	
	2	8	4	4	14	10	15	16	
	3	9	3	4	12	10	14	16	
	4	11	3	4	11	10	13	16	
	5	12	2	3	8	9	11	16	
	6	16	1	3	7	8	11	16	
38	1	3	4	4	14	19	11	7	
	2	11	3	3	14	17	10	7	
	3	12	3	3	13	17	9	6	
	4	16	3	2	10	15	9	6	
	5	18	3	2	9	14	8	5	
	6	20	3	2	9	14	7	5	
39	1	1	1	3	13	18	9	17	
	2	2	1	2	12	16	9	15	
	3	6	1	2	12	16	8	13	
	4	8	1	1	12	15	8	13	
	5	14	1	1	12	13	8	12	
	6	15	1	1	12	13	7	10	
40	1	1	4	5	8	9	10	17	
	2	6	3	5	8	9	9	15	
	3	9	3	5	7	7	6	13	
	4	15	2	5	5	5	6	12	
	5	17	2	5	4	2	4	10	
	6	18	1	5	4	1	1	9	
41	1	2	3	5	9	10	14	16	
	2	4	3	5	8	10	11	13	
	3	8	3	5	7	9	10	10	
	4	12	3	5	6	9	7	10	
	5	13	3	5	5	9	6	7	
	6	20	3	5	5	8	5	5	
42	1	3	2	1	11	19	15	17	
	2	7	2	1	10	15	15	15	
	3	10	2	1	8	14	15	14	
	4	17	2	1	7	11	14	13	
	5	18	2	1	7	10	14	11	
	6	19	2	1	5	8	14	10	
43	1	4	3	3	20	10	15	9	
	2	6	3	2	19	10	13	8	
	3	12	3	2	17	9	9	7	
	4	14	3	2	17	8	8	6	
	5	15	2	1	15	6	6	5	
	6	19	2	1	15	5	3	4	
44	1	2	5	3	16	15	18	5	
	2	8	3	3	14	13	15	5	
	3	11	3	3	14	11	15	5	
	4	12	2	3	13	10	13	5	
	5	17	2	3	12	7	11	5	
	6	19	1	3	11	6	10	5	
45	1	3	5	4	9	14	7	10	
	2	14	4	4	8	13	5	9	
	3	15	4	4	7	12	5	9	
	4	16	4	3	5	12	4	8	
	5	18	3	2	5	12	3	8	
	6	19	3	2	4	11	2	8	
46	1	6	4	2	8	15	15	19	
	2	7	4	2	7	15	15	17	
	3	9	3	2	6	15	15	17	
	4	10	3	2	5	14	14	15	
	5	13	2	2	5	13	14	14	
	6	14	1	2	4	13	14	14	
47	1	1	5	3	8	17	11	19	
	2	12	5	3	7	14	11	19	
	3	13	5	3	6	11	11	18	
	4	14	5	3	6	7	11	18	
	5	15	5	3	6	5	10	18	
	6	18	5	3	5	4	10	17	
48	1	10	4	3	2	17	14	1	
	2	12	4	3	2	16	14	2	
	3	13	4	3	2	16	14	1	
	4	18	3	3	2	15	14	1	
	5	18	2	2	2	15	14	2	
	6	20	2	2	2	15	14	1	
49	1	6	5	1	4	12	13	11	
	2	8	5	1	3	11	12	10	
	3	13	5	1	3	8	11	10	
	4	14	5	1	3	6	9	9	
	5	16	5	1	1	5	7	9	
	6	17	5	1	1	3	5	9	
50	1	1	3	1	14	18	14	8	
	2	2	3	1	13	18	14	7	
	3	5	3	1	12	16	14	6	
	4	6	2	1	11	15	14	5	
	5	7	2	1	9	15	14	3	
	6	17	2	1	9	14	14	3	
51	1	2	2	3	17	11	9	7	
	2	4	2	3	15	11	9	7	
	3	12	2	3	13	9	8	7	
	4	15	2	3	8	7	6	6	
	5	16	1	3	8	5	6	6	
	6	20	1	3	5	4	5	5	
52	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2	N 3	N 4
	36	36	454	414	471	411

************************************************************************