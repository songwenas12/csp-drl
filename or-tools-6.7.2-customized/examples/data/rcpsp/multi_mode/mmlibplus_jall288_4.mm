jobs  (incl. supersource/sink ):	52
RESOURCES
- renewable                 : 4 R
- nonrenewable              : 2 N
- doubly constrained        : 0 D
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
1	1	10		2 3 4 5 6 7 10 11 17 18 
2	9	7		25 24 19 16 14 13 9 
3	9	5		27 22 19 16 8 
4	9	7		38 30 24 22 21 19 9 
5	9	6		27 22 19 15 14 13 
6	9	5		30 25 14 12 9 
7	9	6		38 30 23 21 19 9 
8	9	5		51 29 26 20 14 
9	9	5		39 33 27 26 15 
10	9	7		39 38 33 28 27 24 23 
11	9	7		51 40 37 33 28 25 23 
12	9	2		33 15 
13	9	7		51 40 39 32 29 28 23 
14	9	4		40 38 28 21 
15	9	8		51 49 40 37 32 31 29 28 
16	9	7		51 40 37 36 33 26 23 
17	9	6		40 38 37 32 28 26 
18	9	6		51 38 32 29 28 26 
19	9	5		40 37 32 28 26 
20	9	6		49 40 34 32 30 28 
21	9	8		50 49 39 37 36 35 33 31 
22	9	8		49 47 46 42 41 40 37 33 
23	9	7		50 49 48 46 35 34 31 
24	9	7		51 50 47 46 40 35 34 
25	9	7		50 49 46 45 38 35 34 
26	9	5		49 48 35 34 31 
27	9	3		48 35 29 
28	9	5		50 47 44 36 35 
29	9	5		47 46 45 42 34 
30	9	3		43 37 35 
31	9	4		47 45 42 41 
32	9	4		47 46 42 41 
33	9	2		45 34 
34	9	2		44 43 
35	9	2		42 41 
36	9	2		46 43 
37	9	2		48 44 
38	9	2		43 41 
39	9	2		42 41 
40	9	1		45 
41	9	1		52 
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
2	1	1	3	5	2	5	23	27	
	2	3	3	4	2	4	22	24	
	3	5	3	4	2	4	21	22	
	4	8	3	4	2	4	20	19	
	5	9	3	4	2	4	20	17	
	6	11	3	3	2	4	19	16	
	7	13	3	3	2	4	18	15	
	8	17	3	3	2	4	17	13	
	9	24	3	3	2	4	17	10	
3	1	2	3	1	3	4	17	25	
	2	12	2	1	2	4	16	24	
	3	13	2	1	2	4	15	22	
	4	19	2	1	2	4	13	20	
	5	20	1	1	2	4	10	18	
	6	21	1	1	2	4	7	14	
	7	23	1	1	2	4	5	12	
	8	26	1	1	2	4	3	10	
	9	29	1	1	2	4	3	8	
4	1	4	5	4	5	4	9	21	
	2	7	5	4	5	3	8	19	
	3	8	5	4	5	3	8	18	
	4	12	5	4	5	3	8	16	
	5	13	5	4	5	2	8	15	
	6	19	5	4	5	2	8	13	
	7	23	5	4	5	2	8	12	
	8	27	5	4	5	2	8	11	
	9	28	5	4	5	2	8	10	
5	1	8	3	5	1	3	7	10	
	2	9	3	4	1	3	6	9	
	3	10	3	4	1	3	6	8	
	4	11	3	3	1	2	5	6	
	5	13	3	2	1	2	5	6	
	6	14	2	2	1	2	4	5	
	7	20	2	2	1	1	4	4	
	8	28	2	1	1	1	3	2	
	9	30	2	1	1	1	3	1	
6	1	8	2	5	4	3	23	8	
	2	12	2	4	4	2	22	6	
	3	13	2	4	4	2	20	5	
	4	15	2	4	3	2	18	4	
	5	21	1	4	3	1	16	4	
	6	22	1	3	2	1	12	3	
	7	25	1	3	2	1	11	3	
	8	26	1	3	1	1	10	2	
	9	30	1	3	1	1	6	1	
7	1	5	4	4	5	5	9	14	
	2	10	4	4	5	4	8	11	
	3	11	4	4	5	4	7	11	
	4	12	3	4	5	4	7	8	
	5	21	2	3	5	3	6	7	
	6	22	2	3	5	3	6	6	
	7	23	2	3	5	3	5	3	
	8	28	1	3	5	2	5	2	
	9	30	1	3	5	2	5	1	
8	1	5	5	4	2	3	13	15	
	2	6	4	4	1	3	11	13	
	3	7	4	4	1	3	11	12	
	4	13	3	3	1	3	10	11	
	5	14	2	3	1	2	10	11	
	6	15	2	2	1	2	9	10	
	7	16	2	1	1	2	9	9	
	8	24	1	1	1	2	8	9	
	9	27	1	1	1	2	7	8	
9	1	7	3	5	4	5	30	22	
	2	8	3	5	3	4	28	21	
	3	9	3	5	3	4	28	19	
	4	14	3	5	2	3	28	19	
	5	15	3	5	2	3	27	18	
	6	19	3	5	2	3	27	17	
	7	25	3	5	1	2	26	16	
	8	26	3	5	1	2	25	15	
	9	27	3	5	1	2	25	13	
10	1	3	5	5	5	2	17	12	
	2	4	4	4	4	1	14	12	
	3	13	4	4	4	1	13	11	
	4	14	4	4	4	1	12	9	
	5	15	4	4	3	1	11	9	
	6	16	4	4	3	1	10	8	
	7	17	4	4	3	1	9	6	
	8	25	4	4	2	1	6	6	
	9	28	4	4	2	1	6	5	
11	1	6	4	4	4	4	21	14	
	2	7	3	3	4	3	21	12	
	3	8	3	3	4	3	21	11	
	4	9	2	3	4	3	21	11	
	5	14	2	3	4	2	20	10	
	6	15	2	2	4	2	20	9	
	7	22	2	2	4	2	20	8	
	8	24	1	2	4	2	20	7	
	9	25	1	2	4	2	20	5	
12	1	4	2	5	3	2	24	19	
	2	9	1	4	3	2	21	17	
	3	10	1	4	3	2	20	16	
	4	16	1	4	3	2	18	14	
	5	17	1	4	3	2	16	12	
	6	18	1	3	2	2	13	12	
	7	22	1	3	2	2	11	9	
	8	27	1	3	2	2	9	9	
	9	30	1	3	2	2	8	7	
13	1	9	4	5	5	3	17	12	
	2	10	4	4	4	3	15	11	
	3	14	4	4	4	3	15	10	
	4	16	4	4	4	2	14	10	
	5	17	4	4	3	2	12	8	
	6	23	4	4	3	2	11	7	
	7	25	4	4	3	1	11	7	
	8	28	4	4	3	1	9	5	
	9	29	4	4	3	1	8	5	
14	1	1	4	4	3	4	29	28	
	2	9	4	3	3	4	28	25	
	3	12	4	3	3	4	26	21	
	4	17	4	3	3	4	26	17	
	5	18	3	2	3	4	25	16	
	6	20	3	2	3	4	23	11	
	7	21	2	2	3	4	23	8	
	8	25	2	2	3	4	22	5	
	9	26	2	2	3	4	20	1	
15	1	1	1	4	3	2	13	20	
	2	3	1	3	3	1	12	19	
	3	4	1	3	3	1	12	18	
	4	7	1	3	3	1	10	17	
	5	8	1	3	2	1	8	17	
	6	11	1	2	2	1	8	16	
	7	16	1	2	2	1	6	16	
	8	18	1	2	2	1	5	15	
	9	24	1	2	2	1	4	15	
16	1	5	2	3	1	5	23	25	
	2	11	1	3	1	4	20	25	
	3	13	1	3	1	3	17	21	
	4	15	1	3	1	3	15	19	
	5	25	1	2	1	3	15	18	
	6	27	1	2	1	2	12	15	
	7	28	1	1	1	2	10	13	
	8	29	1	1	1	1	7	11	
	9	30	1	1	1	1	6	11	
17	1	1	1	5	3	3	15	21	
	2	2	1	4	3	3	15	18	
	3	4	1	3	3	3	14	18	
	4	8	1	3	3	2	13	17	
	5	9	1	2	2	2	10	15	
	6	23	1	2	2	2	10	13	
	7	26	1	1	2	1	8	12	
	8	27	1	1	1	1	7	12	
	9	28	1	1	1	1	6	11	
18	1	9	4	5	4	2	24	25	
	2	12	4	4	3	2	22	23	
	3	16	4	4	3	2	18	21	
	4	21	4	4	3	2	15	20	
	5	23	4	3	3	2	15	17	
	6	24	4	3	3	2	10	17	
	7	25	4	3	3	2	10	14	
	8	26	4	2	3	2	5	11	
	9	30	4	2	3	2	4	11	
19	1	2	5	1	4	5	27	15	
	2	5	4	1	3	4	27	14	
	3	11	4	1	3	4	22	14	
	4	13	4	1	2	3	18	14	
	5	15	4	1	2	3	17	14	
	6	28	3	1	2	3	12	13	
	7	29	3	1	1	2	9	13	
	8	30	3	1	1	2	9	12	
	9	30	3	1	1	2	4	13	
20	1	1	4	3	3	3	30	14	
	2	6	4	2	3	3	27	12	
	3	11	4	2	3	3	25	10	
	4	16	4	2	3	3	24	9	
	5	17	4	1	2	3	20	9	
	6	18	3	1	2	3	19	7	
	7	19	3	1	2	3	18	4	
	8	22	3	1	2	3	14	3	
	9	30	3	1	2	3	14	2	
21	1	11	4	5	4	2	26	28	
	2	15	3	4	3	2	22	26	
	3	16	3	4	3	2	20	25	
	4	19	2	4	3	2	20	24	
	5	20	2	4	3	2	18	23	
	6	21	2	3	3	2	14	22	
	7	22	2	3	3	2	11	21	
	8	28	1	3	3	2	10	18	
	9	29	1	3	3	2	8	18	
22	1	1	2	3	2	4	25	16	
	2	6	1	2	2	4	24	16	
	3	8	1	2	2	4	23	12	
	4	11	1	2	2	4	21	12	
	5	15	1	1	2	4	19	9	
	6	21	1	1	2	4	18	9	
	7	24	1	1	2	4	17	6	
	8	27	1	1	2	4	15	3	
	9	29	1	1	2	4	13	3	
23	1	5	5	5	5	4	12	30	
	2	6	5	4	4	3	12	29	
	3	13	5	4	4	3	12	28	
	4	16	5	3	4	3	12	28	
	5	17	5	3	4	3	12	27	
	6	19	5	3	4	3	12	26	
	7	21	5	3	4	3	12	25	
	8	24	5	2	4	3	12	26	
	9	29	5	2	4	3	12	25	
24	1	4	2	4	5	3	4	22	
	2	5	1	4	4	3	4	19	
	3	11	1	4	3	3	3	18	
	4	12	1	4	3	3	3	17	
	5	14	1	4	3	3	3	16	
	6	19	1	3	2	3	2	15	
	7	23	1	3	2	3	2	14	
	8	28	1	3	1	3	1	14	
	9	29	1	3	1	3	1	13	
25	1	1	5	4	4	3	29	30	
	2	3	4	4	4	3	28	28	
	3	14	4	3	4	3	28	27	
	4	16	3	3	4	3	28	26	
	5	17	2	3	4	3	27	23	
	6	19	2	2	3	3	27	23	
	7	22	2	2	3	3	27	20	
	8	27	1	1	3	3	27	19	
	9	28	1	1	3	3	27	18	
26	1	3	2	4	4	4	24	6	
	2	4	2	3	3	4	22	6	
	3	5	2	3	3	4	19	6	
	4	7	2	3	3	4	17	5	
	5	10	2	2	2	4	14	5	
	6	17	1	2	2	4	12	4	
	7	28	1	2	2	4	10	3	
	8	29	1	2	1	4	7	3	
	9	30	1	2	1	4	3	3	
27	1	10	4	4	3	3	16	29	
	2	12	4	3	3	3	16	26	
	3	14	4	3	3	3	14	23	
	4	20	3	3	3	3	14	21	
	5	21	3	3	3	2	12	18	
	6	26	3	3	2	2	11	15	
	7	27	3	3	2	2	11	13	
	8	28	2	3	2	1	9	11	
	9	29	2	3	2	1	8	6	
28	1	1	3	4	4	3	15	2	
	2	2	2	3	3	3	15	1	
	3	5	2	3	3	3	12	1	
	4	7	2	2	3	3	12	1	
	5	8	2	2	2	3	9	1	
	6	10	1	2	2	3	9	1	
	7	11	1	2	2	3	6	1	
	8	19	1	1	1	3	6	1	
	9	20	1	1	1	3	3	1	
29	1	4	2	3	4	4	16	23	
	2	7	1	3	3	4	16	23	
	3	10	1	3	3	4	16	22	
	4	11	1	3	3	4	16	21	
	5	12	1	2	2	4	15	19	
	6	16	1	2	2	3	15	18	
	7	18	1	1	2	3	15	18	
	8	21	1	1	1	3	15	17	
	9	28	1	1	1	3	15	16	
30	1	2	3	3	2	4	5	20	
	2	3	3	3	2	4	4	19	
	3	11	3	3	2	4	4	18	
	4	12	3	3	2	4	3	18	
	5	13	3	2	2	3	3	18	
	6	17	3	2	2	3	3	17	
	7	20	3	2	2	2	2	17	
	8	24	3	2	2	2	2	16	
	9	27	3	2	2	2	2	15	
31	1	5	4	5	1	4	21	24	
	2	7	3	5	1	4	19	23	
	3	11	3	5	1	3	18	22	
	4	14	3	5	1	3	16	21	
	5	17	2	5	1	3	14	20	
	6	19	2	5	1	2	13	19	
	7	25	2	5	1	1	13	17	
	8	27	2	5	1	1	11	17	
	9	28	2	5	1	1	10	16	
32	1	1	4	4	4	5	19	24	
	2	2	4	4	3	4	18	22	
	3	8	4	4	3	4	17	21	
	4	13	3	3	2	4	17	20	
	5	15	3	3	2	3	15	18	
	6	16	2	2	2	3	15	16	
	7	17	2	1	1	3	15	15	
	8	20	1	1	1	3	13	12	
	9	27	1	1	1	3	13	11	
33	1	1	5	5	1	4	2	25	
	2	3	4	4	1	4	2	24	
	3	6	3	4	1	4	2	21	
	4	15	3	3	1	3	2	19	
	5	16	3	3	1	3	2	17	
	6	17	2	2	1	2	2	14	
	7	22	2	1	1	2	2	13	
	8	25	1	1	1	1	2	10	
	9	29	1	1	1	1	2	9	
34	1	1	5	2	3	5	30	24	
	2	12	4	2	3	4	28	20	
	3	14	4	2	3	4	25	17	
	4	21	3	2	3	4	25	13	
	5	23	3	2	3	4	24	11	
	6	24	3	2	3	3	21	11	
	7	25	3	2	3	3	21	7	
	8	26	2	2	3	3	18	3	
	9	27	2	2	3	3	18	1	
35	1	7	4	1	5	4	9	26	
	2	12	4	1	4	4	9	25	
	3	13	3	1	4	4	9	23	
	4	14	3	1	4	4	9	22	
	5	15	2	1	3	4	9	21	
	6	16	2	1	3	3	9	21	
	7	23	2	1	3	3	9	19	
	8	24	1	1	2	3	9	19	
	9	25	1	1	2	3	9	17	
36	1	4	4	3	5	4	26	26	
	2	16	4	3	5	3	24	23	
	3	17	4	3	5	3	22	20	
	4	20	4	2	5	3	21	15	
	5	22	4	2	5	2	21	14	
	6	23	4	2	5	2	19	9	
	7	26	4	2	5	2	19	8	
	8	27	4	1	5	1	18	3	
	9	28	4	1	5	1	16	1	
37	1	1	3	5	4	5	15	23	
	2	2	2	5	4	4	13	23	
	3	3	2	5	4	4	11	22	
	4	4	2	5	4	4	9	21	
	5	16	2	5	4	3	8	20	
	6	17	2	5	4	3	7	18	
	7	19	2	5	4	3	5	17	
	8	22	2	5	4	3	3	15	
	9	23	2	5	4	3	2	14	
38	1	2	4	4	4	2	25	9	
	2	3	3	4	3	2	24	9	
	3	4	3	4	3	2	24	8	
	4	5	3	4	3	2	23	9	
	5	10	3	4	2	2	23	9	
	6	17	3	4	2	1	22	9	
	7	22	3	4	2	1	22	8	
	8	24	3	4	1	1	22	9	
	9	26	3	4	1	1	21	9	
39	1	2	5	4	2	4	26	23	
	2	5	4	4	2	3	22	21	
	3	6	3	4	2	3	22	19	
	4	8	3	4	2	3	19	16	
	5	12	3	4	2	2	18	15	
	6	19	2	3	1	2	15	13	
	7	20	1	3	1	1	12	10	
	8	25	1	3	1	1	10	10	
	9	29	1	3	1	1	10	8	
40	1	1	3	3	4	2	17	23	
	2	4	3	3	4	2	16	21	
	3	6	3	3	3	2	16	20	
	4	7	3	3	3	2	16	18	
	5	12	3	2	3	2	15	17	
	6	13	2	2	2	2	15	15	
	7	14	2	1	1	2	14	14	
	8	23	2	1	1	2	14	11	
	9	24	2	1	1	2	14	10	
41	1	3	4	1	5	5	15	4	
	2	7	4	1	4	4	15	3	
	3	8	4	1	4	4	13	3	
	4	9	4	1	3	4	13	3	
	5	10	3	1	2	3	11	2	
	6	16	3	1	2	3	10	2	
	7	23	3	1	1	3	10	2	
	8	24	3	1	1	3	9	2	
	9	30	3	1	1	3	7	2	
42	1	1	5	3	5	5	27	27	
	2	4	4	2	4	4	26	23	
	3	5	4	2	4	3	25	22	
	4	8	4	2	3	3	23	21	
	5	11	3	2	2	3	22	18	
	6	12	3	2	2	2	21	18	
	7	19	2	2	1	2	18	14	
	8	25	2	2	1	1	18	12	
	9	29	2	2	1	1	17	11	
43	1	6	4	1	4	3	30	7	
	2	11	4	1	4	3	28	6	
	3	14	4	1	4	3	26	6	
	4	15	4	1	4	3	24	5	
	5	17	4	1	4	3	23	5	
	6	22	4	1	4	3	22	5	
	7	23	4	1	4	3	22	4	
	8	26	4	1	4	3	19	4	
	9	29	4	1	4	3	18	4	
44	1	10	4	4	3	5	26	26	
	2	12	4	4	3	4	26	21	
	3	18	4	4	3	4	23	20	
	4	19	3	4	3	3	23	18	
	5	20	3	4	3	3	20	14	
	6	22	3	4	3	3	20	12	
	7	26	3	4	3	2	19	9	
	8	27	2	4	3	2	17	5	
	9	28	2	4	3	2	16	5	
45	1	4	2	5	4	4	14	27	
	2	10	2	5	3	3	12	25	
	3	18	2	5	3	3	12	24	
	4	19	2	5	3	2	11	24	
	5	20	2	5	3	2	11	23	
	6	21	2	5	2	2	10	23	
	7	24	2	5	2	2	10	22	
	8	25	2	5	2	1	10	21	
	9	26	2	5	2	1	9	21	
46	1	1	5	3	5	3	25	7	
	2	2	5	2	4	3	23	6	
	3	3	5	2	4	3	21	6	
	4	4	5	2	4	3	18	5	
	5	8	5	1	4	3	14	4	
	6	9	5	1	4	2	11	4	
	7	21	5	1	4	2	11	3	
	8	24	5	1	4	2	8	3	
	9	30	5	1	4	2	5	3	
47	1	4	5	4	1	3	29	20	
	2	5	4	4	1	3	27	19	
	3	9	4	4	1	3	27	18	
	4	13	3	3	1	3	24	17	
	5	18	3	3	1	3	23	17	
	6	19	3	3	1	3	23	16	
	7	20	2	2	1	3	22	16	
	8	22	2	2	1	3	20	15	
	9	29	2	2	1	3	18	15	
48	1	4	1	5	5	5	22	23	
	2	7	1	5	4	4	21	23	
	3	10	1	5	4	4	17	23	
	4	11	1	5	4	4	17	22	
	5	12	1	5	4	4	14	23	
	6	13	1	5	4	4	10	23	
	7	14	1	5	4	4	9	23	
	8	26	1	5	4	4	6	23	
	9	30	1	5	4	4	3	23	
49	1	4	3	5	1	3	26	23	
	2	6	3	4	1	2	23	22	
	3	7	3	4	1	2	20	19	
	4	11	3	4	1	2	20	16	
	5	12	3	3	1	2	15	16	
	6	16	3	3	1	1	14	14	
	7	26	3	2	1	1	12	11	
	8	27	3	2	1	1	9	9	
	9	29	3	2	1	1	7	8	
50	1	1	5	5	1	3	28	9	
	2	7	4	5	1	3	25	8	
	3	13	4	5	1	3	21	7	
	4	16	4	5	1	3	20	7	
	5	20	4	5	1	3	16	5	
	6	21	3	5	1	3	13	5	
	7	24	3	5	1	3	9	5	
	8	28	3	5	1	3	8	3	
	9	30	3	5	1	3	5	3	
51	1	1	4	1	3	3	25	23	
	2	3	4	1	3	3	24	23	
	3	4	4	1	3	3	22	22	
	4	5	4	1	3	3	21	21	
	5	6	4	1	3	3	18	21	
	6	11	4	1	3	3	17	20	
	7	15	4	1	3	3	15	20	
	8	16	4	1	3	3	14	19	
	9	24	4	1	3	3	13	18	
52	1	0	0	0	0	0	0	0	
************************************************************************

 RESOURCE AVAILABILITIES 
	R 1	R 2	R 3	R 4	N 1	N 2
	26	33	28	27	874	849

************************************************************************
