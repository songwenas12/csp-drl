jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	10		2 3 4 7 8 9 10 13 14 17 
2	3	5		21 16 15 12 5 
3	3	6		36 26 24 21 12 11 
4	3	9		36 32 31 24 22 21 20 16 15 
5	3	1		6 
6	3	8		51 36 32 26 23 20 19 18 
7	3	8		51 36 26 24 23 20 19 18 
8	3	5		23 21 18 16 15 
9	3	3		36 32 11 
10	3	5		51 36 28 23 18 
11	3	7		51 35 33 28 25 23 20 
12	3	4		51 31 22 18 
13	3	4		36 28 21 18 
14	3	9		51 36 35 31 30 29 28 27 26 
15	3	8		38 35 33 30 29 28 27 25 
16	3	7		51 38 33 30 29 28 25 
17	3	7		51 48 38 35 30 27 25 
18	3	6		38 33 30 29 27 25 
19	3	8		49 48 47 38 35 34 28 27 
20	3	6		48 46 38 34 30 27 
21	3	5		48 38 35 30 27 
22	3	6		50 40 38 34 33 28 
23	3	9		49 47 45 43 42 41 38 37 31 
24	3	10		50 49 46 45 44 43 42 41 40 37 
25	3	9		49 46 44 43 42 41 40 37 34 
26	3	7		50 49 47 43 41 38 37 
27	3	6		45 44 43 42 41 39 
28	3	5		45 43 42 41 37 
29	3	5		50 47 44 42 40 
30	3	4		50 47 40 37 
31	3	4		48 46 40 39 
32	3	4		43 42 41 40 
33	3	3		49 48 41 
34	3	2		45 39 
35	3	2		46 40 
36	3	2		47 46 
37	3	1		39 
38	3	1		44 
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
jobnr.	mode	dur	R1	R2	N1	N2	
------------------------------------------------------------------------
1	1	0	0	0	0	0	
2	1	7	10	9	9	0	
	2	8	7	7	6	0	
	3	9	3	7	5	0	
3	1	7	6	4	2	0	
	2	8	5	4	2	0	
	3	9	5	4	0	4	
4	1	1	3	8	4	0	
	2	6	3	6	4	0	
	3	8	3	6	3	0	
5	1	1	5	5	4	0	
	2	5	5	5	3	0	
	3	10	1	5	3	0	
6	1	7	6	7	6	0	
	2	8	6	4	4	0	
	3	10	6	4	3	0	
7	1	5	7	3	0	7	
	2	5	7	3	5	0	
	3	7	5	2	0	5	
8	1	3	8	9	0	9	
	2	4	7	7	0	6	
	3	8	7	2	6	0	
9	1	3	5	9	0	6	
	2	4	4	6	0	4	
	3	9	3	4	6	0	
10	1	3	4	7	0	7	
	2	4	4	7	0	6	
	3	9	4	7	0	2	
11	1	2	9	9	0	6	
	2	8	9	7	7	0	
	3	9	8	3	0	5	
12	1	5	3	6	6	0	
	2	7	2	3	0	4	
	3	9	1	2	0	4	
13	1	1	9	8	0	6	
	2	3	8	6	0	4	
	3	10	8	2	4	0	
14	1	1	9	2	0	9	
	2	4	7	2	6	0	
	3	7	7	2	0	4	
15	1	1	7	7	0	6	
	2	2	3	5	6	0	
	3	4	2	5	4	0	
16	1	3	8	8	0	4	
	2	8	8	6	0	2	
	3	10	8	4	4	0	
17	1	1	8	8	8	0	
	2	2	6	5	0	5	
	3	7	6	4	3	0	
18	1	1	7	3	2	0	
	2	2	7	1	0	6	
	3	10	7	1	0	3	
19	1	2	6	7	6	0	
	2	6	4	7	0	3	
	3	9	4	6	4	0	
20	1	3	8	4	0	4	
	2	5	5	3	0	4	
	3	10	5	3	0	3	
21	1	2	9	5	7	0	
	2	4	7	4	6	0	
	3	8	6	3	0	7	
22	1	2	2	10	7	0	
	2	7	1	8	6	0	
	3	10	1	7	4	0	
23	1	4	3	7	0	7	
	2	6	2	5	0	7	
	3	10	2	4	7	0	
24	1	4	8	8	7	0	
	2	6	7	6	6	0	
	3	7	6	6	4	0	
25	1	4	8	6	0	1	
	2	5	8	5	7	0	
	3	6	6	5	7	0	
26	1	4	7	5	3	0	
	2	6	7	5	2	0	
	3	8	6	3	3	0	
27	1	5	5	7	0	9	
	2	10	4	7	0	8	
	3	10	4	6	5	0	
28	1	3	8	5	0	8	
	2	7	5	3	0	8	
	3	8	5	2	0	8	
29	1	1	6	3	0	7	
	2	5	6	2	2	0	
	3	6	5	2	0	1	
30	1	2	8	4	7	0	
	2	5	6	2	0	8	
	3	6	6	2	0	7	
31	1	5	6	5	2	0	
	2	6	6	5	0	8	
	3	8	6	4	2	0	
32	1	7	6	4	3	0	
	2	8	6	3	0	3	
	3	9	6	2	0	1	
33	1	2	3	10	0	9	
	2	7	2	9	6	0	
	3	8	2	9	0	3	
34	1	5	7	4	0	6	
	2	8	6	3	0	4	
	3	8	6	3	5	0	
35	1	1	3	4	6	0	
	2	9	1	4	0	7	
	3	10	1	4	0	6	
36	1	1	8	4	0	10	
	2	3	7	3	0	9	
	3	7	7	3	6	0	
37	1	2	9	2	0	7	
	2	6	8	2	0	5	
	3	7	8	2	0	3	
38	1	2	3	7	6	0	
	2	3	2	3	3	0	
	3	6	2	2	2	0	
39	1	2	8	9	6	0	
	2	5	7	9	5	0	
	3	6	7	8	4	0	
40	1	5	3	8	8	0	
	2	6	3	8	7	0	
	3	8	3	6	0	6	
41	1	3	5	7	6	0	
	2	4	4	6	5	0	
	3	10	3	2	5	0	
42	1	6	7	5	6	0	
	2	7	7	4	0	3	
	3	8	6	4	0	1	
43	1	1	3	8	0	6	
	2	6	2	6	0	6	
	3	9	2	4	3	0	
44	1	3	4	3	8	0	
	2	7	4	3	0	3	
	3	10	4	3	0	2	
45	1	1	9	7	0	4	
	2	2	6	3	0	2	
	3	3	4	1	4	0	
46	1	2	5	10	6	0	
	2	7	4	9	6	0	
	3	8	4	9	3	0	
47	1	4	6	5	0	9	
	2	10	6	5	4	0	
	3	10	2	5	0	8	
48	1	1	2	10	0	9	
	2	2	1	9	0	8	
	3	3	1	9	0	7	
49	1	3	10	6	8	0	
	2	5	8	6	5	0	
	3	6	8	2	0	4	
50	1	6	7	3	0	6	
	2	8	6	3	0	4	
	3	10	5	2	0	4	
51	1	2	8	7	0	7	
	2	5	8	4	7	0	
	3	5	8	2	0	2	
52	1	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2
	39	38	141	134

************************************************************************
