jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 4 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	18		2 3 4 5 6 7 8 9 10 11 12 13 17 18 21 22 29 30 
2	3	8		50 48 37 28 27 23 20 14 
3	3	9		50 48 45 35 31 27 26 23 19 
4	3	7		42 35 33 28 24 20 15 
5	3	11		51 48 47 45 44 40 39 35 32 26 19 
6	3	4		37 33 19 16 
7	3	10		48 46 45 44 40 35 34 32 27 26 
8	3	9		48 46 45 42 35 34 33 31 27 
9	3	7		51 47 45 36 31 28 26 
10	3	9		50 48 45 44 42 36 35 32 28 
11	3	8		48 46 45 44 41 34 27 26 
12	3	7		47 44 40 38 33 32 24 
13	3	8		50 48 46 44 43 38 36 31 
14	3	6		51 45 44 40 32 26 
15	3	6		50 48 44 41 36 27 
16	3	4		48 42 32 25 
17	3	7		45 44 42 40 39 38 32 
18	3	4		47 42 34 33 
19	3	4		49 46 43 34 
20	3	4		46 44 41 34 
21	3	4		45 44 39 35 
22	3	4		42 40 38 32 
23	3	4		44 40 39 38 
24	3	4		49 48 45 39 
25	3	4		47 40 39 38 
26	3	3		42 38 33 
27	3	3		47 39 38 
28	3	3		46 40 39 
29	3	3		49 46 44 
30	3	3		43 42 40 
31	3	2		40 32 
32	3	1		41 
33	3	1		43 
34	3	1		38 
35	3	1		38 
36	3	1		39 
37	3	1		45 
38	3	1		52 
39	3	1		52 
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
jobnr.	mode	dur	R1	R2	N1	N2	N3	N4	
------------------------------------------------------------------------
1	1	0	0	0	0	0	0	0	
2	1	3	3	8	1	10	9	10	
	2	7	3	8	1	8	7	6	
	3	10	3	8	1	7	5	3	
3	1	3	2	8	8	9	1	4	
	2	4	1	5	7	4	1	4	
	3	10	1	4	7	3	1	3	
4	1	6	1	4	9	6	7	4	
	2	9	1	4	9	4	6	4	
	3	10	1	1	8	3	6	3	
5	1	4	5	3	6	8	6	5	
	2	6	2	3	5	6	6	3	
	3	8	2	1	2	4	6	1	
6	1	7	9	9	9	9	5	8	
	2	8	7	4	9	7	4	8	
	3	10	3	2	8	7	1	7	
7	1	3	6	6	4	9	4	7	
	2	6	5	4	4	8	3	6	
	3	10	4	4	4	8	2	5	
8	1	3	7	6	9	2	8	6	
	2	5	5	4	8	1	7	4	
	3	6	5	3	7	1	5	3	
9	1	5	8	5	6	4	5	7	
	2	6	8	5	4	2	5	5	
	3	10	6	3	3	1	5	1	
10	1	1	7	7	8	5	6	9	
	2	8	5	3	5	3	3	7	
	3	9	4	1	2	2	1	7	
11	1	7	10	9	3	6	5	6	
	2	8	8	4	3	5	4	5	
	3	9	6	2	2	5	4	5	
12	1	4	9	3	9	7	4	5	
	2	7	9	3	8	3	3	5	
	3	9	9	1	6	1	3	3	
13	1	4	7	4	7	8	8	9	
	2	8	4	4	6	8	5	5	
	3	10	4	2	5	7	4	4	
14	1	3	7	2	6	6	9	7	
	2	4	6	1	6	6	7	5	
	3	6	5	1	6	6	4	3	
15	1	4	9	6	7	4	8	10	
	2	8	8	5	3	3	4	9	
	3	10	6	5	2	2	3	9	
16	1	3	10	10	7	5	7	8	
	2	4	6	6	6	5	6	7	
	3	9	4	5	2	5	5	6	
17	1	5	7	5	4	6	7	3	
	2	6	5	5	2	4	7	2	
	3	7	5	5	2	4	7	1	
18	1	1	7	5	5	6	5	10	
	2	2	6	3	3	5	5	7	
	3	10	4	2	2	5	5	5	
19	1	3	7	4	9	3	4	5	
	2	6	6	4	7	3	4	5	
	3	10	3	3	3	3	4	3	
20	1	2	5	9	2	7	9	4	
	2	3	2	8	1	6	9	2	
	3	9	2	8	1	5	9	1	
21	1	1	4	6	8	2	9	8	
	2	2	3	6	7	1	5	6	
	3	3	3	3	6	1	4	5	
22	1	1	4	6	8	7	9	8	
	2	2	3	6	7	6	6	8	
	3	10	2	5	7	6	5	4	
23	1	4	6	7	10	7	6	6	
	2	9	5	7	10	6	6	5	
	3	10	3	7	10	5	5	2	
24	1	1	9	5	4	8	4	7	
	2	6	7	3	2	8	4	6	
	3	9	6	3	2	8	3	5	
25	1	6	6	9	3	5	7	5	
	2	9	4	9	3	5	4	3	
	3	10	3	8	1	4	4	3	
26	1	5	10	6	5	4	9	7	
	2	7	10	5	3	3	6	6	
	3	9	10	3	3	2	5	5	
27	1	2	10	6	2	5	7	7	
	2	8	9	4	2	4	7	4	
	3	10	9	3	2	4	5	3	
28	1	1	7	8	7	5	9	4	
	2	7	3	3	7	4	3	2	
	3	9	3	1	7	4	3	2	
29	1	1	10	9	8	7	7	2	
	2	7	8	4	6	5	7	2	
	3	10	7	2	3	4	4	1	
30	1	1	7	9	8	8	5	4	
	2	6	4	6	8	7	4	3	
	3	10	1	3	5	5	4	2	
31	1	3	4	9	4	9	5	9	
	2	9	4	5	4	8	5	7	
	3	10	4	4	4	7	5	4	
32	1	6	4	6	7	7	8	6	
	2	8	4	6	4	5	7	3	
	3	9	3	6	2	5	3	2	
33	1	4	9	6	5	9	9	5	
	2	5	8	5	5	8	4	5	
	3	8	7	5	5	7	3	5	
34	1	7	6	6	7	5	6	4	
	2	8	5	5	7	4	4	4	
	3	10	4	5	7	4	2	4	
35	1	1	5	7	10	8	5	1	
	2	4	4	7	7	8	2	1	
	3	6	3	5	7	6	1	1	
36	1	4	9	8	10	4	7	6	
	2	8	5	8	8	3	3	5	
	3	9	4	7	5	2	1	3	
37	1	1	9	10	2	3	2	7	
	2	3	7	9	1	2	2	4	
	3	8	5	8	1	2	2	1	
38	1	3	4	8	3	10	7	10	
	2	4	2	7	2	9	6	9	
	3	8	2	6	1	8	6	9	
39	1	2	6	8	7	9	7	4	
	2	6	5	5	7	8	6	3	
	3	9	5	3	7	7	6	2	
40	1	6	6	10	8	8	8	7	
	2	8	6	7	6	8	5	5	
	3	9	6	3	4	8	4	4	
41	1	5	4	5	10	8	8	6	
	2	6	4	5	8	7	6	6	
	3	8	2	5	8	6	6	6	
42	1	7	5	7	1	3	8	8	
	2	9	2	7	1	2	7	6	
	3	10	1	7	1	1	7	6	
43	1	5	8	7	5	10	7	7	
	2	9	7	5	5	8	6	5	
	3	10	6	1	5	8	5	3	
44	1	1	8	7	6	7	7	10	
	2	2	3	6	4	7	7	5	
	3	10	3	3	2	7	7	3	
45	1	1	2	9	5	7	4	3	
	2	3	1	8	4	7	3	3	
	3	6	1	5	4	5	2	3	
46	1	1	4	8	10	5	6	10	
	2	2	4	4	8	5	5	5	
	3	9	4	2	8	1	5	3	
47	1	5	8	5	2	8	4	2	
	2	6	5	2	2	5	3	2	
	3	10	4	1	2	4	2	2	
48	1	4	10	5	10	8	7	3	
	2	5	6	5	9	7	5	2	
	3	8	6	4	9	7	4	2	
49	1	2	8	9	3	6	8	6	
	2	7	8	7	3	5	4	6	
	3	10	8	7	3	4	4	6	
50	1	1	8	7	2	6	10	6	
	2	3	6	5	2	5	8	4	
	3	9	5	5	2	4	7	4	
51	1	1	6	7	2	8	10	5	
	2	2	6	7	2	5	6	5	
	3	8	6	3	2	4	4	4	
52	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2	N 3	N 4
	61	61	277	301	301	278

************************************************************************
