jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 4 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	4		2 3 4 10 
2	3	7		17 14 13 12 9 8 7 
3	3	4		15 8 7 5 
4	3	2		19 6 
5	3	5		20 19 14 13 11 
6	3	3		18 17 9 
7	3	4		20 19 18 11 
8	3	4		24 20 18 11 
9	3	3		24 20 11 
10	3	5		25 24 22 21 20 
11	3	3		22 21 16 
12	3	3		25 21 18 
13	3	3		32 24 18 
14	3	2		21 16 
15	3	3		25 23 21 
16	3	4		32 27 25 23 
17	3	4		29 26 22 21 
18	3	4		29 28 23 22 
19	3	6		35 33 32 31 29 27 
20	3	3		33 27 23 
21	3	3		35 32 27 
22	3	4		35 33 31 30 
23	3	2		31 26 
24	3	2		33 27 
25	3	3		34 33 29 
26	3	4		37 35 34 30 
27	3	2		30 28 
28	3	6		43 40 38 37 36 34 
29	3	2		37 30 
30	3	6		43 42 41 40 38 36 
31	3	4		38 37 36 34 
32	3	4		43 40 38 34 
33	3	4		51 41 38 36 
34	3	4		51 42 41 39 
35	3	3		41 38 36 
36	3	2		49 39 
37	3	2		49 39 
38	3	1		39 
39	3	5		50 47 46 45 44 
40	3	5		51 50 49 46 45 
41	3	4		49 47 46 45 
42	3	3		48 46 45 
43	3	2		49 47 
44	3	1		48 
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
jobnr.	mode	dur	R1	R2	N1	N2	N3	N4	
------------------------------------------------------------------------
1	1	0	0	0	0	0	0	0	
2	1	2	8	2	1	9	7	6	
	2	4	7	2	1	5	7	5	
	3	8	5	2	1	5	5	4	
3	1	5	2	7	5	5	6	4	
	2	6	2	4	4	3	6	4	
	3	7	2	2	4	1	4	2	
4	1	1	2	3	6	4	6	2	
	2	2	1	3	5	3	5	2	
	3	8	1	3	4	2	5	1	
5	1	3	7	5	10	5	6	5	
	2	6	6	5	8	2	4	5	
	3	9	6	4	7	2	3	5	
6	1	4	8	3	5	7	5	7	
	2	5	7	2	4	4	5	6	
	3	10	7	1	4	2	5	2	
7	1	1	5	10	3	8	6	9	
	2	3	3	8	3	6	5	8	
	3	10	2	7	3	3	4	8	
8	1	5	8	5	9	5	6	9	
	2	7	7	4	7	4	6	7	
	3	8	7	4	5	4	5	4	
9	1	1	10	8	8	7	5	7	
	2	7	5	8	7	6	2	6	
	3	8	1	6	7	5	2	5	
10	1	2	3	9	10	6	8	7	
	2	9	1	6	10	6	6	7	
	3	10	1	4	10	5	5	5	
11	1	1	6	6	9	8	8	7	
	2	2	5	4	8	5	5	7	
	3	7	5	1	7	4	3	6	
12	1	2	4	6	6	8	8	7	
	2	6	3	4	5	6	8	5	
	3	10	3	3	5	5	5	5	
13	1	5	10	5	7	9	10	10	
	2	9	5	4	6	6	9	5	
	3	10	2	4	6	6	8	4	
14	1	2	9	10	9	8	9	9	
	2	3	8	8	9	7	7	8	
	3	5	7	8	8	7	6	8	
15	1	2	6	8	7	4	3	7	
	2	5	6	6	7	3	3	7	
	3	9	6	5	6	2	2	7	
16	1	1	4	10	8	10	7	7	
	2	4	2	10	7	7	4	5	
	3	5	2	10	7	6	3	3	
17	1	1	5	8	5	5	1	4	
	2	3	4	6	4	4	1	2	
	3	5	3	4	3	3	1	1	
18	1	1	7	4	7	6	8	3	
	2	2	6	2	6	5	7	3	
	3	4	6	1	6	5	7	3	
19	1	5	7	6	1	5	9	9	
	2	6	3	6	1	3	7	8	
	3	7	3	4	1	3	1	8	
20	1	3	5	4	10	9	4	6	
	2	8	4	4	7	7	4	6	
	3	10	4	4	5	5	4	6	
21	1	3	8	2	8	5	6	1	
	2	4	6	2	8	4	6	2	
	3	5	6	2	8	4	6	1	
22	1	7	4	3	6	8	10	7	
	2	9	4	3	6	8	7	5	
	3	10	4	3	6	5	6	5	
23	1	4	6	2	7	3	10	8	
	2	5	6	2	4	3	7	8	
	3	8	5	2	3	2	5	8	
24	1	2	5	6	8	4	4	8	
	2	7	3	6	7	3	3	7	
	3	10	3	6	4	2	3	7	
25	1	8	9	9	7	8	10	4	
	2	9	8	6	3	7	7	4	
	3	10	7	2	2	7	4	4	
26	1	6	4	8	10	6	7	8	
	2	7	3	7	9	6	6	6	
	3	8	1	7	7	6	6	4	
27	1	5	3	8	7	6	9	10	
	2	7	2	6	6	5	8	8	
	3	8	2	4	5	4	8	6	
28	1	1	7	7	9	6	4	7	
	2	8	6	5	8	5	3	5	
	3	10	4	5	5	5	2	4	
29	1	7	1	10	6	9	2	3	
	2	9	1	6	6	8	2	3	
	3	10	1	4	6	8	1	3	
30	1	4	9	7	4	5	10	5	
	2	6	9	7	3	4	9	4	
	3	8	9	7	1	3	8	1	
31	1	1	8	4	10	7	9	8	
	2	5	4	4	7	4	6	4	
	3	9	3	2	5	4	5	1	
32	1	5	8	5	7	8	8	7	
	2	8	6	4	4	5	8	4	
	3	10	6	4	4	4	7	4	
33	1	1	8	9	7	7	6	10	
	2	2	7	7	6	4	4	10	
	3	3	4	6	5	2	2	10	
34	1	2	6	6	5	5	10	8	
	2	4	5	6	4	3	7	6	
	3	6	4	5	4	3	7	6	
35	1	7	7	8	6	7	7	9	
	2	8	6	6	4	7	6	8	
	3	9	4	4	2	5	5	8	
36	1	3	9	7	8	7	7	5	
	2	6	9	7	7	6	6	5	
	3	9	7	7	6	6	4	5	
37	1	1	9	3	1	6	9	8	
	2	9	9	2	1	5	6	6	
	3	10	9	1	1	5	4	5	
38	1	1	4	8	9	2	8	6	
	2	4	4	7	6	2	5	4	
	3	5	1	7	5	2	2	3	
39	1	2	2	10	8	7	9	6	
	2	5	2	9	7	6	9	5	
	3	7	2	8	7	4	9	5	
40	1	2	6	7	8	9	4	9	
	2	9	6	4	7	9	3	8	
	3	10	6	4	6	9	1	8	
41	1	3	4	5	1	8	8	10	
	2	4	3	4	1	7	8	9	
	3	10	1	2	1	7	6	7	
42	1	7	7	10	4	6	3	5	
	2	8	4	9	4	6	1	2	
	3	9	2	8	2	5	1	1	
43	1	5	9	7	2	7	6	7	
	2	6	9	5	2	6	6	6	
	3	8	9	4	1	5	3	6	
44	1	1	7	3	9	8	8	10	
	2	5	6	2	8	6	8	7	
	3	7	6	2	7	6	8	6	
45	1	5	5	5	10	10	3	4	
	2	7	4	5	10	6	3	4	
	3	10	3	5	10	2	3	2	
46	1	2	4	7	5	2	7	2	
	2	6	3	7	4	2	7	2	
	3	10	3	5	4	2	7	1	
47	1	3	7	9	9	3	9	5	
	2	4	3	6	7	3	7	5	
	3	10	3	5	5	3	4	5	
48	1	1	10	9	8	8	4	6	
	2	2	7	8	8	5	3	5	
	3	7	3	6	4	2	2	4	
49	1	1	6	1	10	5	8	4	
	2	2	5	1	7	4	5	2	
	3	7	5	1	3	4	4	2	
50	1	1	7	7	8	6	5	4	
	2	2	5	5	5	3	5	2	
	3	3	5	5	1	2	5	2	
51	1	7	7	6	7	8	6	3	
	2	8	6	5	6	8	5	3	
	3	9	5	5	2	8	2	1	
52	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2	N 3	N 4
	32	34	312	295	308	297

************************************************************************
