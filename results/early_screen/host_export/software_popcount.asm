
/tmp/collatz-host-export-bench:     file format elf64-x86-64


Disassembly of section .text:

000000000001a4e0 <software_popcount>:
   1a4e0:	48 85 f6             	test   %rsi,%rsi
   1a4e3:	74 0f                	je     1a4f4 <software_popcount+0x14>
   1a4e5:	48 83 fe 04          	cmp    $0x4,%rsi
   1a4e9:	73 10                	jae    1a4fb <software_popcount+0x1b>
   1a4eb:	31 c9                	xor    %ecx,%ecx
   1a4ed:	31 c0                	xor    %eax,%eax
   1a4ef:	e9 01 01 00 00       	jmp    1a5f5 <software_popcount+0x115>
   1a4f4:	31 c0                	xor    %eax,%eax
   1a4f6:	e9 f4 00 00 00       	jmp    1a5ef <software_popcount+0x10f>
   1a4fb:	48 89 f1             	mov    %rsi,%rcx
   1a4fe:	48 83 e1 fc          	and    $0xfffffffffffffffc,%rcx
   1a502:	66 0f ef c0          	pxor   %xmm0,%xmm0
   1a506:	31 c0                	xor    %eax,%eax
   1a508:	66 0f 6f 15 e0 ef fe 	movdqa -0x11020(%rip),%xmm2        # 94f0 <GCC_except_table379+0xe4>
   1a50f:	ff 
   1a510:	66 0f 6f 1d 18 f1 fe 	movdqa -0x10ee8(%rip),%xmm3        # 9630 <GCC_except_table379+0x224>
   1a517:	ff 
   1a518:	66 0f 6f 2d 90 ef fe 	movdqa -0x11070(%rip),%xmm5        # 94b0 <GCC_except_table379+0xa4>
   1a51f:	ff 
   1a520:	66 0f ef e4          	pxor   %xmm4,%xmm4
   1a524:	66 0f ef c9          	pxor   %xmm1,%xmm1
   1a528:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
   1a52f:	00 
   1a530:	f3 0f 7e 3c 87       	movq   (%rdi,%rax,4),%xmm7
   1a535:	f3 0f 7e 74 87 08    	movq   0x8(%rdi,%rax,4),%xmm6
   1a53b:	66 0f 62 f8          	punpckldq %xmm0,%xmm7
   1a53f:	66 44 0f 6f c7       	movdqa %xmm7,%xmm8
   1a544:	66 41 0f 71 d0 01    	psrlw  $0x1,%xmm8
   1a54a:	66 44 0f db c2       	pand   %xmm2,%xmm8
   1a54f:	66 41 0f f8 f8       	psubb  %xmm8,%xmm7
   1a554:	66 44 0f 6f c7       	movdqa %xmm7,%xmm8
   1a559:	66 44 0f db c3       	pand   %xmm3,%xmm8
   1a55e:	66 0f 71 d7 02       	psrlw  $0x2,%xmm7
   1a563:	66 0f db fb          	pand   %xmm3,%xmm7
   1a567:	66 41 0f fc f8       	paddb  %xmm8,%xmm7
   1a56c:	66 44 0f 6f c7       	movdqa %xmm7,%xmm8
   1a571:	66 41 0f 71 d0 04    	psrlw  $0x4,%xmm8
   1a577:	66 44 0f fc c7       	paddb  %xmm7,%xmm8
   1a57c:	66 44 0f db c5       	pand   %xmm5,%xmm8
   1a581:	66 44 0f f6 c0       	psadbw %xmm0,%xmm8
   1a586:	66 41 0f d4 e0       	paddq  %xmm8,%xmm4
   1a58b:	66 0f 62 f0          	punpckldq %xmm0,%xmm6
   1a58f:	66 0f 6f fe          	movdqa %xmm6,%xmm7
   1a593:	66 0f 71 d7 01       	psrlw  $0x1,%xmm7
   1a598:	66 0f db fa          	pand   %xmm2,%xmm7
   1a59c:	66 0f f8 f7          	psubb  %xmm7,%xmm6
   1a5a0:	66 0f 6f fe          	movdqa %xmm6,%xmm7
   1a5a4:	66 0f db fb          	pand   %xmm3,%xmm7
   1a5a8:	66 0f 71 d6 02       	psrlw  $0x2,%xmm6
   1a5ad:	66 0f db f3          	pand   %xmm3,%xmm6
   1a5b1:	66 0f fc f7          	paddb  %xmm7,%xmm6
   1a5b5:	66 0f 6f fe          	movdqa %xmm6,%xmm7
   1a5b9:	66 0f 71 d7 04       	psrlw  $0x4,%xmm7
   1a5be:	66 0f fc fe          	paddb  %xmm6,%xmm7
   1a5c2:	66 0f db fd          	pand   %xmm5,%xmm7
   1a5c6:	66 0f f6 f8          	psadbw %xmm0,%xmm7
   1a5ca:	66 0f d4 cf          	paddq  %xmm7,%xmm1
   1a5ce:	48 83 c0 04          	add    $0x4,%rax
   1a5d2:	48 39 c1             	cmp    %rax,%rcx
   1a5d5:	0f 85 55 ff ff ff    	jne    1a530 <software_popcount+0x50>
   1a5db:	66 0f d4 cc          	paddq  %xmm4,%xmm1
   1a5df:	66 0f 70 c1 ee       	pshufd $0xee,%xmm1,%xmm0
   1a5e4:	66 0f d4 c1          	paddq  %xmm1,%xmm0
   1a5e8:	66 48 0f 7e c0       	movq   %xmm0,%rax
   1a5ed:	eb 01                	jmp    1a5f0 <software_popcount+0x110>
   1a5ef:	c3                   	ret
   1a5f0:	48 39 ce             	cmp    %rcx,%rsi
   1a5f3:	74 fa                	je     1a5ef <software_popcount+0x10f>
   1a5f5:	8b 14 8f             	mov    (%rdi,%rcx,4),%edx
   1a5f8:	41 89 d0             	mov    %edx,%r8d
   1a5fb:	41 d1 e8             	shr    $1,%r8d
   1a5fe:	41 81 e0 55 55 55 55 	and    $0x55555555,%r8d
   1a605:	44 29 c2             	sub    %r8d,%edx
   1a608:	41 89 d0             	mov    %edx,%r8d
   1a60b:	41 81 e0 33 33 33 33 	and    $0x33333333,%r8d
   1a612:	c1 ea 02             	shr    $0x2,%edx
   1a615:	81 e2 33 33 33 33    	and    $0x33333333,%edx
   1a61b:	44 01 c2             	add    %r8d,%edx
   1a61e:	41 89 d0             	mov    %edx,%r8d
   1a621:	41 c1 e8 04          	shr    $0x4,%r8d
   1a625:	41 01 d0             	add    %edx,%r8d
   1a628:	41 81 e0 0f 0f 0f 0f 	and    $0xf0f0f0f,%r8d
   1a62f:	41 69 d0 01 01 01 01 	imul   $0x1010101,%r8d,%edx
   1a636:	c1 ea 18             	shr    $0x18,%edx
   1a639:	48 01 d0             	add    %rdx,%rax
   1a63c:	48 ff c1             	inc    %rcx
   1a63f:	eb af                	jmp    1a5f0 <software_popcount+0x110>

Disassembly of section .init:

Disassembly of section .fini:

Disassembly of section .plt:
