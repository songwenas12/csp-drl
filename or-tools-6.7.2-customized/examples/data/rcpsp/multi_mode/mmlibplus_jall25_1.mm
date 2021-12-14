jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 2 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	7		2 3 4 5 6 7 17 
2	3	3		10 9 8 
3	3	5		18 16 14 12 10 
4	3	3		20 11 8 
5	3	4		18 14 12 10 
6	3	4		18 16 12 10 
7	3	6		24 18 16 14 12 11 
8	3	6		24 18 16 14 13 12 
9	3	5		22 20 18 13 12 
10	3	3		22 20 11 
11	3	3		19 15 13 
12	3	2		19 15 
13	3	4		27 25 23 21 
14	3	1		15 
15	3	4		36 27 23 21 
16	3	4		33 26 25 22 
17	3	3		29 25 21 
18	3	6		36 33 30 29 27 26 
19	3	5		33 30 29 26 25 
20	3	5		33 30 29 27 24 
21	3	3		33 30 26 
22	3	3		30 29 28 
23	3	4		42 32 30 29 
24	3	2		31 26 
25	3	3		36 34 28 
26	3	2		34 28 
27	3	2		34 28 
28	3	6		44 42 39 37 35 32 
29	3	5		44 43 41 37 34 
30	3	2		44 31 
31	3	4		43 41 39 35 
32	3	4		43 41 40 38 
33	3	4		43 41 40 38 
34	3	2		39 38 
35	3	2		40 38 
36	3	4		50 49 48 40 
37	3	1		38 
38	3	6		50 49 48 47 46 45 
39	3	3		51 48 40 
40	3	3		47 46 45 
41	3	3		51 46 45 
42	3	3		48 47 45 
43	3	1		45 
44	3	1		46 
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
2	1	4	9	4	4	9	
	2	7	5	3	3	8	
	3	8	5	3	2	6	
3	1	7	6	3	6	10	
	2	9	6	3	4	6	
	3	10	6	2	4	6	
4	1	3	8	8	5	5	
	2	5	8	8	4	4	
	3	7	6	7	4	3	
5	1	2	6	1	7	3	
	2	3	5	1	7	3	
	3	4	2	1	7	3	
6	1	3	8	6	8	6	
	2	5	7	5	6	5	
	3	9	4	4	5	3	
7	1	5	3	1	5	3	
	2	6	2	1	3	2	
	3	7	2	1	2	2	
8	1	5	6	9	10	2	
	2	7	4	9	7	2	
	3	10	2	9	7	2	
9	1	2	3	3	7	7	
	2	7	3	3	6	4	
	3	8	3	3	2	2	
10	1	1	8	8	2	10	
	2	7	4	7	1	9	
	3	9	4	6	1	9	
11	1	2	3	2	9	7	
	2	8	3	1	9	7	
	3	10	3	1	9	5	
12	1	2	8	4	9	4	
	2	3	7	2	7	2	
	3	6	7	1	4	2	
13	1	3	8	6	4	9	
	2	4	5	6	2	7	
	3	5	2	6	2	5	
14	1	1	8	4	7	10	
	2	8	7	4	4	10	
	3	9	7	1	3	10	
15	1	6	3	9	6	5	
	2	7	3	8	6	4	
	3	10	3	8	6	3	
16	1	1	2	5	6	7	
	2	3	1	5	5	6	
	3	5	1	3	3	5	
17	1	1	10	3	6	7	
	2	4	6	1	6	5	
	3	8	3	1	4	5	
18	1	7	7	3	5	6	
	2	9	5	2	4	4	
	3	10	2	2	4	4	
19	1	1	10	7	7	8	
	2	3	9	5	7	3	
	3	5	7	4	7	2	
20	1	6	2	6	8	2	
	2	8	2	2	8	1	
	3	9	2	1	7	1	
21	1	4	7	9	5	1	
	2	5	5	9	4	1	
	3	8	4	7	3	1	
22	1	6	6	2	7	9	
	2	7	5	2	4	9	
	3	8	4	1	3	9	
23	1	3	7	3	10	5	
	2	8	4	3	10	5	
	3	9	2	3	10	5	
24	1	5	9	3	8	3	
	2	8	8	2	5	3	
	3	9	6	2	1	3	
25	1	1	9	5	8	9	
	2	7	6	5	8	5	
	3	9	4	4	8	3	
26	1	2	9	7	8	5	
	2	3	4	5	6	4	
	3	10	2	2	6	3	
27	1	2	8	4	8	5	
	2	8	8	3	6	3	
	3	10	7	2	6	3	
28	1	1	7	8	6	9	
	2	4	5	8	4	6	
	3	10	5	8	3	6	
29	1	4	5	8	7	8	
	2	5	2	6	5	6	
	3	9	2	6	5	4	
30	1	1	4	4	7	10	
	2	5	4	4	5	7	
	3	8	4	3	5	3	
31	1	1	8	9	9	2	
	2	2	5	8	6	2	
	3	9	3	6	2	2	
32	1	3	7	7	9	7	
	2	4	5	6	6	6	
	3	5	3	5	4	5	
33	1	1	8	5	5	5	
	2	7	7	5	4	5	
	3	8	6	3	4	3	
34	1	5	8	2	7	6	
	2	7	6	2	6	5	
	3	8	5	2	5	3	
35	1	1	2	7	7	9	
	2	2	2	7	3	3	
	3	9	2	6	3	1	
36	1	2	10	4	3	7	
	2	3	7	4	2	6	
	3	10	3	1	2	4	
37	1	1	4	5	5	2	
	2	5	3	4	4	1	
	3	8	2	2	3	1	
38	1	1	8	9	8	7	
	2	2	8	6	5	6	
	3	4	8	2	2	4	
39	1	7	7	8	9	4	
	2	8	7	6	7	4	
	3	9	7	3	4	4	
40	1	1	6	8	9	7	
	2	2	3	8	7	6	
	3	6	3	8	6	3	
41	1	3	6	6	2	5	
	2	4	5	4	2	4	
	3	10	3	3	2	4	
42	1	3	7	7	8	6	
	2	7	7	6	7	5	
	3	10	7	5	6	1	
43	1	1	9	5	3	6	
	2	9	9	3	3	5	
	3	10	9	3	3	4	
44	1	7	4	6	10	6	
	2	9	4	5	7	3	
	3	10	4	4	7	3	
45	1	5	5	4	7	8	
	2	8	3	3	6	7	
	3	10	3	3	5	5	
46	1	2	3	9	6	7	
	2	7	3	9	6	5	
	3	9	3	9	4	5	
47	1	1	9	8	6	9	
	2	5	5	6	3	9	
	3	9	3	4	1	9	
48	1	5	5	4	2	9	
	2	6	3	4	2	8	
	3	9	3	3	2	8	
49	1	3	5	4	8	8	
	2	8	4	2	5	5	
	3	9	4	2	4	4	
50	1	6	5	8	7	8	
	2	7	5	6	6	8	
	3	8	3	3	4	7	
51	1	1	5	10	4	7	
	2	6	4	9	4	7	
	3	9	4	9	4	6	
52	1	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	N 1	N 2
	40	28	239	232

************************************************************************
