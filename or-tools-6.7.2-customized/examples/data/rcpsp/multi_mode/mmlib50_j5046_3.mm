jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	8		2 3 4 5 6 7 8 9 
2	3	8		32 24 23 22 20 16 15 11 
3	3	8		38 32 23 22 19 18 13 12 
4	3	6		38 22 14 13 12 10 
5	3	4		32 20 11 10 
6	3	6		22 20 18 14 13 10 
7	3	6		38 32 23 19 13 12 
8	3	6		32 30 24 22 18 13 
9	3	11		37 34 33 30 29 25 24 23 22 21 18 
10	3	9		36 34 33 31 25 24 23 19 17 
11	3	3		18 14 13 
12	3	6		34 33 30 24 20 17 
13	3	6		36 34 33 26 25 17 
14	3	4		36 33 30 17 
15	3	9		50 49 38 36 33 31 30 29 21 
16	3	8		50 49 36 35 34 29 26 21 
17	3	8		50 49 47 45 37 35 29 21 
18	3	9		50 49 47 46 39 36 35 31 27 
19	3	5		50 47 35 30 21 
20	3	6		51 49 47 35 31 26 
21	3	4		46 39 28 27 
22	3	8		48 47 46 45 44 41 40 39 
23	3	6		46 45 44 43 39 35 
24	3	4		48 47 46 26 
25	3	5		45 43 41 40 39 
26	3	4		43 41 40 39 
27	3	4		48 43 42 40 
28	3	4		44 43 42 41 
29	3	4		51 44 43 41 
30	3	4		46 45 42 40 
31	3	4		45 44 41 40 
32	3	4		49 46 42 40 
33	3	3		47 44 40 
34	3	3		47 45 43 
35	3	2		41 40 
36	3	2		45 44 
37	3	2		46 40 
38	3	2		41 40 
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
jobnr.	mode	dur	R1	R2	N1	N2	
------------------------------------------------------------------------
1	1	0	0	0	0	0	
2	1	4	8	8	8	5	
	2	5	8	6	7	5	
	3	9	7	3	6	4	
3	1	3	1	8	9	3	
	2	6	1	7	9	2	
	3	10	1	3	9	2	
4	1	4	7	9	9	6	
	2	6	4	7	8	6	
	3	8	2	5	8	6	
5	1	6	3	9	7	4	
	2	7	2	7	5	4	
	3	9	2	5	5	4	
6	1	1	4	10	7	3	
	2	2	3	8	4	3	
	3	10	1	6	2	3	
7	1	6	7	3	7	6	
	2	7	6	3	4	5	
	3	10	3	3	3	4	
8	1	2	6	5	10	9	
	2	3	4	5	9	7	
	3	7	2	5	8	7	
9	1	2	6	6	9	8	
	2	4	6	4	3	8	
	3	5	5	2	2	7	
10	1	3	4	8	10	5	
	2	4	4	4	10	4	
	3	5	4	1	10	3	
11	1	1	8	10	2	4	
	2	5	6	10	2	3	
	3	7	6	10	2	1	
12	1	3	9	8	8	1	
	2	7	7	7	6	1	
	3	9	4	6	6	1	
13	1	6	3	8	4	4	
	2	7	2	8	3	2	
	3	10	2	7	3	2	
14	1	1	5	4	2	7	
	2	3	4	2	2	7	
	3	5	3	2	2	7	
15	1	8	8	7	3	6	
	2	9	8	5	2	4	
	3	10	6	1	1	4	
16	1	2	7	5	7	6	
	2	6	5	3	5	4	
	3	8	1	1	5	3	
17	1	2	5	6	6	7	
	2	8	3	4	6	7	
	3	10	1	4	2	6	
18	1	4	8	5	4	5	
	2	6	5	5	4	4	
	3	7	4	5	4	3	
19	1	6	10	6	6	8	
	2	9	9	5	5	8	
	3	10	9	5	5	7	
20	1	2	2	3	8	9	
	2	3	2	3	8	8	
	3	10	2	3	7	8	
21	1	2	8	3	5	9	
	2	4	7	3	5	7	
	3	9	7	3	3	6	
22	1	5	2	2	8	7	
	2	9	2	1	7	7	
	3	10	2	1	7	6	
23	1	4	8	6	9	6	
	2	6	8	6	9	4	
	3	10	7	5	9	1	
24	1	4	7	10	8	9	
	2	8	4	8	4	9	
	3	10	2	6	2	9	
25	1	1	7	8	7	8	
	2	4	4	7	6	5	
	3	9	3	7	2	5	
26	1	2	10	6	7	1	
	2	8	6	5	5	1	
	3	9	3	4	5	1	
27	1	3	7	7	9	5	
	2	9	3	7	7	4	
	3	10	1	3	3	1	
28	1	5	7	2	9	8	
	2	9	4	2	8	6	
	3	10	3	2	7	6	
29	1	2	8	9	6	6	
	2	9	6	9	4	6	
	3	10	6	7	3	5	
30	1	8	6	7	8	5	
	2	9	4	6	7	5	
	3	10	4	2	4	4	
31	1	6	7	3	3	4	
	2	7	5	3	2	3	
	3	8	4	3	1	3	
32	1	5	7	9	8	8	
	2	6	7	9	8	7	
	3	8	6	9	8	7	
33	1	6	9	7	9	9	
	2	8	6	6	7	7	
	3	9	4	4	6	6	
34	1	1	9	9	6	1	
	2	7	6	4	5	1	
	3	10	6	3	5	1	
35	1	4	7	8	6	10	
	2	5	4	7	4	9	
	3	10	2	4	4	8	
36	1	5	6	6	6	7	
	2	6	5	6	3	3	
	3	8	4	6	2	1	
37	1	4	4	8	8	8	
	2	9	4	8	8	5	
	3	10	3	8	8	4	
38	1	1	6	7	5	4	
	2	5	5	6	3	3	
	3	9	4	6	2	3	
39	1	1	6	8	10	7	
	2	6	6	6	7	6	
	3	8	5	4	3	6	
40	1	6	9	9	8	8	
	2	9	8	9	7	8	
	3	10	8	9	7	7	
41	1	5	3	3	3	7	
	2	8	3	2	3	5	
	3	10	3	1	3	1	
42	1	2	8	7	7	6	
	2	8	5	7	5	4	
	3	9	4	7	4	3	
43	1	8	7	4	6	2	
	2	9	6	3	5	2	
	3	10	2	2	5	2	
44	1	1	2	2	2	8	
	2	8	2	2	1	4	
	3	9	1	2	1	2	
45	1	5	6	8	7	9	
	2	6	3	7	4	8	
	3	8	3	6	3	7	
46	1	5	6	4	6	6	
	2	8	4	2	6	4	
	3	9	3	1	6	2	
47	1	1	5	10	3	6	
	2	4	5	8	2	3	
	3	10	1	8	2	2	
48	1	2	6	7	9	7	
	2	3	5	5	8	6	
	3	8	4	5	8	3	
49	1	3	6	5	6	9	
	2	5	6	5	4	7	
	3	7	6	3	4	7	
50	1	4	5	2	7	7	
	2	6	4	1	7	4	
	3	10	2	1	4	3	
51	1	2	2	7	6	4	
	2	3	2	4	6	3	
	3	4	2	3	5	2	
52	1	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2
	25	25	279	256

************************************************************************
