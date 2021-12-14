jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 4 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	9		2 3 4 6 7 8 9 15 16 
2	9	7		22 21 19 17 13 11 5 
3	9	5		24 23 21 20 10 
4	9	5		20 19 17 13 12 
5	9	6		34 32 28 27 18 14 
6	9	6		34 28 26 25 18 13 
7	9	4		26 25 20 13 
8	9	8		32 31 30 29 27 25 23 19 
9	9	7		32 30 29 25 23 21 19 
10	9	5		30 28 27 17 14 
11	9	6		37 29 27 26 23 20 
12	9	10		51 50 38 37 34 33 32 27 26 25 
13	9	7		37 35 31 30 29 27 23 
14	9	7		38 37 33 31 29 26 25 
15	9	7		37 33 32 31 29 27 24 
16	9	6		51 38 34 32 31 24 
17	9	6		50 46 37 34 33 26 
18	9	5		51 50 49 30 23 
19	9	7		50 48 47 37 35 34 26 
20	9	9		51 50 49 48 47 46 35 34 32 
21	9	8		49 48 47 46 44 39 35 31 
22	9	6		49 41 38 37 34 30 
23	9	5		45 41 39 38 33 
24	9	7		48 47 46 45 44 39 35 
25	9	5		49 46 44 36 35 
26	9	5		49 44 41 39 36 
27	9	6		49 48 46 44 41 40 
28	9	6		51 47 45 44 41 40 
29	9	5		51 46 44 41 39 
30	9	6		48 47 46 45 43 40 
31	9	3		50 43 36 
32	9	4		45 44 43 40 
33	9	4		48 47 44 42 
34	9	3		44 43 40 
35	9	2		41 40 
36	9	2		45 40 
37	9	2		43 40 
38	9	2		47 46 
39	9	1		40 
40	9	1		42 
41	9	1		43 
42	9	1		52 
43	9	1		52 
44	9	1		52 
45	9	1		52 
46	9	1		52 
47	9	1		52 
48	9	1		52 
49	9	1		52 
50	9	1		52 
51	9	1		52 
52	1	0		
************************************************************************
REQUESTS/DURATIONS
jobnr.	mode	dur	R1	R2	R3	R4	N1	N2	
------------------------------------------------------------------------
1	1	0	0	0	0	0	0	0	
2	1	2	3	5	5	2	30	18	
	2	3	2	4	4	2	30	17	
	3	4	2	4	4	2	30	16	
	4	8	2	3	4	2	30	17	
	5	10	2	3	4	2	30	16	
	6	11	2	3	4	2	30	15	
	7	20	2	2	4	2	30	17	
	8	24	2	2	4	2	30	16	
	9	26	2	2	4	2	30	15	
3	1	2	2	5	3	2	15	24	
	2	15	2	4	2	2	14	23	
	3	16	2	3	2	2	13	23	
	4	17	2	3	2	2	11	23	
	5	18	2	3	2	2	10	23	
	6	20	1	2	1	2	8	23	
	7	21	1	1	1	2	6	23	
	8	22	1	1	1	2	5	23	
	9	30	1	1	1	2	3	23	
4	1	4	4	1	4	5	27	22	
	2	5	3	1	3	4	27	20	
	3	6	3	1	3	4	23	19	
	4	7	3	1	3	3	20	19	
	5	8	3	1	2	3	18	18	
	6	13	3	1	2	3	17	17	
	7	21	3	1	1	3	16	16	
	8	26	3	1	1	2	11	15	
	9	28	3	1	1	2	11	14	
5	1	1	4	5	4	1	17	22	
	2	2	3	4	3	1	16	19	
	3	6	3	4	3	1	16	18	
	4	7	3	4	3	1	14	18	
	5	12	3	4	3	1	13	16	
	6	14	3	4	3	1	13	15	
	7	15	3	4	3	1	12	15	
	8	22	3	4	3	1	10	13	
	9	29	3	4	3	1	9	12	
6	1	4	2	5	5	3	28	6	
	2	6	2	5	4	2	27	5	
	3	12	2	5	4	2	26	5	
	4	17	2	5	4	2	25	5	
	5	22	2	5	3	2	25	4	
	6	23	1	5	3	2	25	4	
	7	24	1	5	3	2	24	4	
	8	25	1	5	2	2	23	4	
	9	28	1	5	2	2	23	3	
7	1	1	1	5	5	3	28	13	
	2	11	1	5	5	3	27	10	
	3	13	1	5	5	3	26	10	
	4	14	1	5	5	2	26	9	
	5	16	1	5	5	2	25	7	
	6	17	1	5	5	2	25	5	
	7	18	1	5	5	2	24	5	
	8	19	1	5	5	1	24	3	
	9	20	1	5	5	1	24	1	
8	1	4	4	3	1	4	10	28	
	2	7	4	3	1	3	9	25	
	3	9	4	3	1	3	9	21	
	4	10	4	3	1	3	8	19	
	5	11	4	2	1	2	7	19	
	6	12	3	2	1	2	7	17	
	7	18	3	2	1	2	7	14	
	8	19	3	1	1	2	6	12	
	9	22	3	1	1	2	6	11	
9	1	9	1	4	4	3	12	8	
	2	14	1	4	3	2	10	7	
	3	16	1	4	3	2	10	6	
	4	18	1	3	2	2	9	5	
	5	21	1	3	2	2	8	5	
	6	22	1	2	2	2	8	4	
	7	23	1	2	1	2	7	4	
	8	24	1	1	1	2	7	2	
	9	27	1	1	1	2	6	2	
10	1	6	4	4	4	2	15	21	
	2	8	4	4	3	2	14	21	
	3	9	3	4	3	2	12	18	
	4	10	3	4	3	2	12	17	
	5	11	3	4	3	2	10	17	
	6	12	2	4	3	1	10	14	
	7	14	1	4	3	1	8	12	
	8	25	1	4	3	1	7	12	
	9	26	1	4	3	1	7	11	
11	1	1	2	4	5	4	26	28	
	2	5	1	4	5	4	25	26	
	3	9	1	4	5	4	21	22	
	4	12	1	4	5	4	21	18	
	5	17	1	4	5	3	19	16	
	6	21	1	3	5	3	18	12	
	7	23	1	3	5	2	15	10	
	8	25	1	3	5	2	14	6	
	9	28	1	3	5	2	12	2	
12	1	4	3	2	4	3	27	14	
	2	6	3	2	4	2	24	14	
	3	10	3	2	4	2	23	12	
	4	11	3	2	3	2	20	12	
	5	13	3	1	3	1	18	10	
	6	14	3	1	2	1	16	10	
	7	20	3	1	1	1	14	8	
	8	25	3	1	1	1	11	7	
	9	28	3	1	1	1	9	6	
13	1	1	2	4	4	4	26	28	
	2	2	2	4	3	3	26	28	
	3	5	2	4	3	3	25	26	
	4	6	2	4	3	3	22	26	
	5	13	2	3	3	2	22	24	
	6	15	2	3	3	2	19	24	
	7	16	2	3	3	2	19	23	
	8	20	2	2	3	1	16	21	
	9	24	2	2	3	1	15	21	
14	1	5	3	4	4	4	17	18	
	2	6	3	4	4	4	16	16	
	3	9	3	4	4	4	15	14	
	4	13	3	3	3	3	14	14	
	5	18	3	3	3	2	13	11	
	6	20	3	2	3	2	13	11	
	7	21	3	2	2	1	12	9	
	8	25	3	1	2	1	12	7	
	9	26	3	1	2	1	11	6	
15	1	3	4	5	3	4	14	24	
	2	6	4	4	3	4	11	22	
	3	7	4	4	3	4	9	21	
	4	17	4	4	3	4	9	20	
	5	20	3	3	3	4	8	18	
	6	25	3	3	3	4	6	16	
	7	26	3	3	3	4	5	16	
	8	27	2	2	3	4	3	14	
	9	30	2	2	3	4	1	12	
16	1	3	5	2	2	4	9	7	
	2	4	4	1	2	4	8	5	
	3	21	4	1	2	4	8	4	
	4	23	3	1	2	4	8	5	
	5	25	2	1	2	4	7	4	
	6	27	2	1	1	4	6	3	
	7	28	1	1	1	4	6	3	
	8	29	1	1	1	4	4	3	
	9	30	1	1	1	4	4	2	
17	1	2	4	5	5	3	16	25	
	2	3	4	5	5	3	14	23	
	3	7	4	5	5	3	13	21	
	4	8	4	5	5	3	13	18	
	5	16	4	5	5	2	12	15	
	6	17	4	5	5	2	11	13	
	7	18	4	5	5	2	9	12	
	8	27	4	5	5	2	9	9	
	9	28	4	5	5	2	7	8	
18	1	5	5	3	3	2	21	27	
	2	10	5	2	2	2	21	25	
	3	11	5	2	2	2	21	22	
	4	14	5	2	2	2	21	18	
	5	15	5	2	2	2	20	16	
	6	21	5	2	2	2	20	11	
	7	22	5	2	2	2	20	10	
	8	29	5	2	2	2	19	5	
	9	30	5	2	2	2	19	4	
19	1	8	5	4	4	4	2	21	
	2	9	4	4	4	4	2	20	
	3	12	4	4	4	4	2	17	
	4	14	4	4	4	4	2	16	
	5	15	4	4	3	4	2	12	
	6	19	4	4	3	4	2	10	
	7	21	4	4	3	4	2	8	
	8	25	4	4	3	4	2	5	
	9	29	4	4	3	4	2	3	
20	1	3	3	1	5	5	22	23	
	2	7	3	1	4	4	18	23	
	3	14	3	1	4	4	16	23	
	4	15	3	1	3	4	13	23	
	5	23	2	1	3	4	13	23	
	6	24	2	1	2	3	9	22	
	7	25	1	1	1	3	9	22	
	8	27	1	1	1	3	6	22	
	9	29	1	1	1	3	5	22	
21	1	3	5	4	4	5	16	29	
	2	5	4	4	4	5	15	29	
	3	13	4	3	4	5	14	27	
	4	15	4	3	3	5	11	26	
	5	16	4	3	3	5	8	25	
	6	22	4	2	3	5	8	24	
	7	24	4	2	2	5	6	23	
	8	25	4	1	2	5	2	22	
	9	29	4	1	2	5	1	21	
22	1	4	4	3	2	4	21	26	
	2	10	3	3	2	4	21	25	
	3	22	3	3	2	4	21	23	
	4	24	3	3	2	4	21	22	
	5	25	2	3	2	4	21	22	
	6	26	2	3	2	4	20	21	
	7	27	2	3	2	4	20	20	
	8	29	1	3	2	4	20	18	
	9	30	1	3	2	4	20	17	
23	1	1	5	4	1	2	22	28	
	2	2	4	3	1	2	22	28	
	3	3	4	3	1	2	20	27	
	4	4	3	3	1	2	20	26	
	5	8	3	3	1	1	19	25	
	6	15	3	3	1	1	17	24	
	7	23	3	3	1	1	17	23	
	8	28	2	3	1	1	15	23	
	9	30	2	3	1	1	14	22	
24	1	4	1	1	1	4	22	29	
	2	5	1	1	1	4	19	27	
	3	7	1	1	1	4	17	25	
	4	13	1	1	1	4	16	24	
	5	15	1	1	1	4	12	23	
	6	17	1	1	1	4	11	22	
	7	18	1	1	1	4	9	21	
	8	22	1	1	1	4	7	21	
	9	30	1	1	1	4	6	20	
25	1	1	3	3	1	1	15	15	
	2	8	2	3	1	1	15	15	
	3	9	2	3	1	1	15	14	
	4	11	2	3	1	1	15	13	
	5	17	2	3	1	1	15	12	
	6	19	2	3	1	1	15	11	
	7	22	2	3	1	1	15	10	
	8	27	2	3	1	1	15	9	
	9	30	2	3	1	1	15	8	
26	1	1	5	2	4	3	16	27	
	2	2	4	2	4	3	15	26	
	3	4	4	2	4	3	14	26	
	4	8	3	2	3	3	13	26	
	5	9	3	2	3	3	13	25	
	6	10	3	2	3	3	12	25	
	7	23	3	2	3	3	11	25	
	8	25	2	2	2	3	10	25	
	9	29	2	2	2	3	10	24	
27	1	1	4	4	5	3	26	19	
	2	4	3	3	4	2	24	19	
	3	12	3	3	4	2	23	15	
	4	19	3	3	4	2	22	15	
	5	20	3	2	3	1	21	12	
	6	22	2	2	3	1	20	12	
	7	25	2	2	3	1	20	10	
	8	26	2	2	3	1	18	7	
	9	27	2	2	3	1	17	5	
28	1	4	3	5	3	4	23	21	
	2	7	3	4	3	4	20	18	
	3	16	3	4	3	4	18	17	
	4	21	3	4	3	4	17	17	
	5	25	3	3	2	3	17	14	
	6	26	3	3	2	3	15	12	
	7	27	3	2	2	3	12	12	
	8	29	3	2	1	2	12	9	
	9	30	3	2	1	2	10	8	
29	1	1	4	4	4	3	7	26	
	2	2	4	3	4	3	7	26	
	3	3	4	3	4	3	7	25	
	4	5	3	3	3	3	7	25	
	5	7	2	3	3	3	7	24	
	6	11	2	3	2	3	7	24	
	7	13	1	3	1	3	7	24	
	8	23	1	3	1	3	7	23	
	9	28	1	3	1	3	7	22	
30	1	2	3	3	4	4	12	25	
	2	3	2	2	3	4	11	24	
	3	4	2	2	3	4	10	23	
	4	8	2	2	2	4	9	23	
	5	21	2	1	2	4	8	22	
	6	22	2	1	2	4	6	21	
	7	25	2	1	2	4	6	20	
	8	29	2	1	1	4	5	20	
	9	30	2	1	1	4	3	20	
31	1	4	5	4	2	5	23	28	
	2	5	4	3	2	4	20	27	
	3	11	4	3	2	4	19	27	
	4	13	4	3	2	4	15	25	
	5	21	3	2	1	4	14	24	
	6	23	3	2	1	4	12	24	
	7	24	3	2	1	4	10	23	
	8	26	2	2	1	4	7	22	
	9	27	2	2	1	4	4	21	
32	1	7	4	3	1	2	24	16	
	2	11	4	3	1	1	22	16	
	3	12	4	3	1	1	22	15	
	4	16	4	2	1	1	20	14	
	5	17	4	2	1	1	17	13	
	6	21	4	2	1	1	17	11	
	7	22	4	1	1	1	13	10	
	8	26	4	1	1	1	12	10	
	9	27	4	1	1	1	11	9	
33	1	9	4	4	1	2	1	15	
	2	11	4	4	1	2	1	14	
	3	14	4	4	1	2	1	13	
	4	16	4	4	1	2	1	12	
	5	18	4	3	1	1	1	14	
	6	19	4	3	1	1	1	13	
	7	24	4	3	1	1	1	12	
	8	25	4	2	1	1	1	13	
	9	27	4	2	1	1	1	12	
34	1	1	3	4	3	2	17	27	
	2	2	3	4	2	2	16	27	
	3	4	3	4	2	2	15	26	
	4	6	2	4	2	2	15	26	
	5	8	2	4	2	2	14	25	
	6	10	2	4	2	1	14	25	
	7	14	2	4	2	1	13	25	
	8	17	1	4	2	1	12	24	
	9	22	1	4	2	1	12	23	
35	1	9	4	4	2	3	24	28	
	2	15	3	4	2	2	23	28	
	3	16	3	3	2	2	22	28	
	4	18	3	3	2	2	20	28	
	5	24	3	3	1	2	18	28	
	6	25	2	2	1	1	16	28	
	7	26	2	1	1	1	16	28	
	8	27	2	1	1	1	14	28	
	9	29	2	1	1	1	12	28	
36	1	2	2	4	3	3	5	5	
	2	4	2	4	3	3	4	5	
	3	5	2	4	3	3	4	4	
	4	6	2	3	3	2	4	4	
	5	10	2	2	3	2	3	5	
	6	11	2	2	3	2	3	4	
	7	26	2	2	3	2	3	3	
	8	27	2	1	3	1	3	3	
	9	28	2	1	3	1	3	2	
37	1	3	5	2	4	5	29	26	
	2	4	4	1	4	4	27	25	
	3	7	4	1	4	4	27	22	
	4	8	4	1	4	4	25	16	
	5	10	3	1	3	3	25	13	
	6	11	3	1	3	3	24	13	
	7	12	3	1	3	3	22	7	
	8	28	3	1	3	3	22	4	
	9	29	3	1	3	3	21	4	
38	1	6	5	4	5	4	13	11	
	2	11	4	4	5	3	12	10	
	3	15	4	4	5	3	12	9	
	4	17	3	4	5	3	11	9	
	5	18	3	4	5	2	10	8	
	6	19	3	4	5	2	10	7	
	7	22	3	4	5	1	9	6	
	8	28	2	4	5	1	9	5	
	9	29	2	4	5	1	8	5	
39	1	4	3	4	1	2	19	29	
	2	6	2	4	1	1	17	26	
	3	7	2	4	1	1	17	22	
	4	12	2	4	1	1	16	19	
	5	13	2	3	1	1	15	18	
	6	17	2	3	1	1	14	14	
	7	20	2	2	1	1	12	11	
	8	21	2	2	1	1	12	9	
	9	22	2	2	1	1	11	6	
40	1	5	4	3	5	4	14	19	
	2	6	4	3	4	4	12	19	
	3	7	4	3	4	4	11	19	
	4	9	4	2	4	3	10	19	
	5	10	4	2	4	3	10	18	
	6	16	4	2	3	2	9	19	
	7	24	4	2	3	2	8	19	
	8	27	4	1	3	1	6	20	
	9	28	4	1	3	1	6	19	
41	1	9	5	4	5	3	17	26	
	2	11	4	4	4	2	16	23	
	3	12	4	4	3	2	16	23	
	4	13	4	4	3	2	16	21	
	5	14	4	4	2	2	16	20	
	6	17	3	4	2	2	16	18	
	7	23	3	4	2	2	16	17	
	8	24	3	4	1	2	16	16	
	9	29	3	4	1	2	16	15	
42	1	5	1	4	5	4	19	20	
	2	8	1	4	4	4	16	20	
	3	9	1	4	4	4	16	18	
	4	10	1	4	4	4	13	15	
	5	12	1	4	4	3	12	13	
	6	19	1	4	4	3	10	10	
	7	22	1	4	4	3	8	9	
	8	23	1	4	4	3	5	6	
	9	24	1	4	4	3	3	6	
43	1	12	5	5	4	4	17	30	
	2	13	4	5	4	3	16	29	
	3	14	4	5	4	3	15	28	
	4	15	4	5	3	3	14	27	
	5	17	3	5	3	3	13	25	
	6	18	3	5	3	3	13	24	
	7	22	3	5	3	3	12	23	
	8	23	3	5	2	3	10	23	
	9	26	3	5	2	3	10	22	
44	1	1	4	5	4	3	25	26	
	2	2	4	4	4	3	24	23	
	3	5	4	4	4	3	23	17	
	4	8	4	3	4	3	21	16	
	5	14	4	3	4	3	21	13	
	6	16	4	3	4	3	20	12	
	7	19	4	2	4	3	20	9	
	8	22	4	2	4	3	18	5	
	9	28	4	2	4	3	18	3	
45	1	2	4	1	5	4	24	25	
	2	5	4	1	4	3	23	21	
	3	7	4	1	3	3	22	20	
	4	12	4	1	3	3	19	18	
	5	16	4	1	3	3	18	13	
	6	20	3	1	2	3	17	12	
	7	26	3	1	2	3	16	11	
	8	27	3	1	1	3	13	9	
	9	28	3	1	1	3	12	7	
46	1	3	4	3	4	5	29	16	
	2	5	3	2	4	4	28	16	
	3	13	3	2	4	4	28	15	
	4	19	3	2	4	4	27	14	
	5	21	2	1	3	4	27	12	
	6	22	2	1	3	4	27	11	
	7	24	2	1	2	4	26	11	
	8	28	2	1	2	4	26	10	
	9	29	2	1	2	4	26	9	
47	1	3	1	3	4	3	16	18	
	2	7	1	3	3	3	16	17	
	3	12	1	3	3	3	15	16	
	4	16	1	3	3	3	14	16	
	5	17	1	3	3	3	13	15	
	6	20	1	3	3	2	13	14	
	7	24	1	3	3	2	12	14	
	8	27	1	3	3	2	12	13	
	9	29	1	3	3	2	11	12	
48	1	1	5	4	4	5	4	21	
	2	4	5	4	4	4	3	18	
	3	14	5	4	4	3	3	16	
	4	15	5	4	4	3	3	15	
	5	21	5	4	4	3	3	13	
	6	23	5	4	4	2	3	11	
	7	24	5	4	4	2	3	8	
	8	26	5	4	4	1	3	6	
	9	27	5	4	4	1	3	5	
49	1	3	5	1	3	4	15	17	
	2	4	4	1	2	3	15	16	
	3	8	4	1	2	3	13	16	
	4	14	4	1	2	3	12	16	
	5	21	3	1	1	3	12	15	
	6	24	3	1	1	3	9	15	
	7	28	2	1	1	3	8	15	
	8	29	2	1	1	3	7	15	
	9	30	2	1	1	3	6	15	
50	1	3	4	1	4	4	20	20	
	2	7	4	1	4	3	17	19	
	3	8	4	1	4	3	16	17	
	4	14	3	1	4	2	14	16	
	5	15	3	1	3	2	13	15	
	6	16	3	1	3	2	10	15	
	7	17	2	1	3	1	10	14	
	8	19	2	1	2	1	9	12	
	9	26	2	1	2	1	7	12	
51	1	1	5	4	3	2	29	20	
	2	6	4	3	2	2	28	19	
	3	10	4	3	2	2	27	16	
	4	12	4	3	2	2	27	15	
	5	17	4	3	2	1	26	14	
	6	21	4	3	1	1	26	11	
	7	24	4	3	1	1	25	9	
	8	29	4	3	1	1	25	7	
	9	30	4	3	1	1	25	6	
52	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	R 3	R 4	N 1	N 2
	20	20	19	19	827	949

************************************************************************
