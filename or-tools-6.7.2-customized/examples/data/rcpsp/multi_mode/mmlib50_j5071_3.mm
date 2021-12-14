jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	10		2 3 4 5 6 7 9 10 14 15 
2	3	6		21 20 19 18 16 8 
3	3	3		13 12 11 
4	3	9		35 27 26 25 23 22 21 19 17 
5	3	8		34 28 25 23 22 21 20 18 
6	3	7		35 28 26 22 20 19 13 
7	3	7		27 26 25 23 20 19 18 
8	3	5		35 28 26 25 13 
9	3	8		51 34 29 28 24 23 22 18 
10	3	12		51 39 37 35 34 30 29 28 27 25 23 22 
11	3	9		35 34 33 30 29 24 23 22 21 
12	3	7		34 33 30 28 26 23 20 
13	3	10		51 39 37 34 33 32 31 30 29 24 
14	3	5		39 37 34 33 19 
15	3	7		51 39 34 33 30 29 23 
16	3	9		51 47 39 36 35 34 32 31 30 
17	3	6		51 39 33 32 29 24 
18	3	9		48 47 39 38 37 36 35 33 30 
19	3	6		51 36 32 31 30 29 
20	3	5		51 37 32 29 24 
21	3	9		50 49 47 46 43 42 39 37 31 
22	3	11		50 49 48 47 46 45 44 43 41 38 36 
23	3	8		49 48 47 46 44 43 36 32 
24	3	10		50 49 48 47 46 45 44 43 41 36 
25	3	8		49 48 47 46 45 43 41 36 
26	3	8		50 49 48 47 43 42 41 39 
27	3	8		48 47 46 44 43 42 41 40 
28	3	6		48 47 46 45 41 36 
29	3	7		47 46 45 43 42 41 40 
30	3	6		50 49 43 42 41 40 
31	3	5		48 45 44 41 38 
32	3	4		45 42 41 38 
33	3	4		45 44 41 40 
34	3	4		46 44 43 40 
35	3	4		46 44 43 41 
36	3	2		42 40 
37	3	2		45 40 
38	3	1		40 
39	3	1		44 
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
2	1	2	4	6	9	0	
	2	2	4	5	0	2	
	3	10	2	3	0	2	
3	1	3	7	7	0	5	
	2	5	6	5	9	0	
	3	5	5	3	0	1	
4	1	1	9	10	0	5	
	2	1	6	7	5	0	
	3	10	6	7	0	2	
5	1	1	4	10	8	0	
	2	2	3	6	0	8	
	3	6	2	6	3	0	
6	1	3	7	8	7	0	
	2	5	7	7	0	10	
	3	6	5	6	4	0	
7	1	6	8	7	7	0	
	2	7	7	6	0	5	
	3	9	4	4	0	4	
8	1	1	3	5	9	0	
	2	3	1	4	8	0	
	3	4	1	2	7	0	
9	1	1	4	6	7	0	
	2	1	4	5	0	7	
	3	2	4	4	0	7	
10	1	3	6	6	2	0	
	2	6	6	4	1	0	
	3	7	6	3	0	5	
11	1	5	7	2	8	0	
	2	7	7	2	7	0	
	3	10	7	2	6	0	
12	1	7	7	3	8	0	
	2	9	5	3	0	5	
	3	10	4	2	0	4	
13	1	2	7	6	4	0	
	2	3	6	5	0	4	
	3	8	5	4	3	0	
14	1	4	4	5	0	6	
	2	8	2	4	8	0	
	3	10	1	4	0	6	
15	1	3	10	10	0	2	
	2	5	7	9	0	2	
	3	9	6	7	0	2	
16	1	3	10	7	7	0	
	2	4	7	4	7	0	
	3	8	6	3	7	0	
17	1	6	4	5	0	6	
	2	8	4	3	3	0	
	3	9	4	3	2	0	
18	1	2	6	2	5	0	
	2	7	6	2	4	0	
	3	10	3	1	4	0	
19	1	2	4	4	10	0	
	2	8	3	2	0	5	
	3	9	3	2	0	4	
20	1	2	7	9	7	0	
	2	5	6	6	3	0	
	3	8	4	5	3	0	
21	1	1	7	5	4	0	
	2	4	7	3	4	0	
	3	6	6	1	0	2	
22	1	2	4	6	9	0	
	2	5	3	4	0	5	
	3	7	3	2	7	0	
23	1	4	10	10	3	0	
	2	6	7	9	0	8	
	3	7	6	9	0	7	
24	1	2	9	7	8	0	
	2	4	7	6	7	0	
	3	7	4	5	0	5	
25	1	1	8	6	7	0	
	2	1	6	6	0	9	
	3	8	6	6	0	8	
26	1	3	9	9	0	8	
	2	8	8	8	5	0	
	3	9	8	8	4	0	
27	1	5	7	5	0	8	
	2	6	7	4	5	0	
	3	6	5	2	0	2	
28	1	5	7	6	5	0	
	2	6	5	4	0	4	
	3	9	4	4	0	4	
29	1	4	4	5	6	0	
	2	5	2	3	5	0	
	3	9	1	3	4	0	
30	1	3	9	9	0	5	
	2	9	8	6	0	5	
	3	10	8	6	0	3	
31	1	2	7	3	0	2	
	2	3	6	2	0	2	
	3	6	4	1	7	0	
32	1	5	10	9	8	0	
	2	6	8	8	0	8	
	3	9	4	7	7	0	
33	1	6	8	8	8	0	
	2	6	7	6	0	5	
	3	9	7	4	0	5	
34	1	1	10	2	0	5	
	2	4	8	2	6	0	
	3	7	4	2	0	4	
35	1	7	3	9	10	0	
	2	7	1	8	0	3	
	3	8	1	5	4	0	
36	1	1	9	9	4	0	
	2	7	7	5	0	1	
	3	10	4	2	2	0	
37	1	2	7	10	0	8	
	2	4	4	7	0	8	
	3	8	3	6	4	0	
38	1	4	9	7	8	0	
	2	7	8	6	4	0	
	3	8	7	5	3	0	
39	1	2	8	7	2	0	
	2	5	6	5	2	0	
	3	9	2	3	2	0	
40	1	1	6	8	0	6	
	2	4	5	8	0	5	
	3	6	5	8	0	4	
41	1	1	9	8	9	0	
	2	2	7	5	8	0	
	3	10	6	5	8	0	
42	1	4	4	7	8	0	
	2	7	3	5	0	3	
	3	8	3	3	0	3	
43	1	1	8	8	6	0	
	2	2	6	6	0	8	
	3	6	4	3	0	7	
44	1	3	3	10	0	9	
	2	3	1	8	5	0	
	3	4	1	8	4	0	
45	1	4	8	5	0	6	
	2	4	7	2	6	0	
	3	9	7	2	0	2	
46	1	4	10	10	0	7	
	2	7	6	8	0	5	
	3	9	4	8	4	0	
47	1	4	7	4	7	0	
	2	7	6	4	5	0	
	3	7	1	4	0	1	
48	1	1	6	5	0	7	
	2	8	5	4	3	0	
	3	9	4	3	0	7	
49	1	3	9	8	0	6	
	2	3	9	8	7	0	
	3	4	9	8	0	2	
50	1	4	8	9	8	0	
	2	6	4	8	8	0	
	3	8	3	8	0	6	
51	1	3	9	5	4	0	
	2	4	6	4	4	0	
	3	8	3	4	2	0	
52	1	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2
	60	61	235	167

************************************************************************
