
/tmp/collatz-host-export-bench:     file format elf64-x86-64


Disassembly of section .text:

000000000001a410 <hardware_popcount>:
   1a410:	48 83 fe 08          	cmp    $0x8,%rsi
   1a414:	73 0e                	jae    1a424 <hardware_popcount+0x14>
   1a416:	45 31 c9             	xor    %r9d,%r9d
   1a419:	31 d2                	xor    %edx,%edx
   1a41b:	31 c0                	xor    %eax,%eax
   1a41d:	31 c9                	xor    %ecx,%ecx
   1a41f:	45 31 c0             	xor    %r8d,%r8d
   1a422:	eb 43                	jmp    1a467 <hardware_popcount+0x57>
   1a424:	45 31 d2             	xor    %r10d,%r10d
   1a427:	45 31 c0             	xor    %r8d,%r8d
   1a42a:	31 c9                	xor    %ecx,%ecx
   1a42c:	31 c0                	xor    %eax,%eax
   1a42e:	31 d2                	xor    %edx,%edx
   1a430:	f3 4e 0f b8 0c 97    	popcnt (%rdi,%r10,4),%r9
   1a436:	f3 4e 0f b8 5c 97 08 	popcnt 0x8(%rdi,%r10,4),%r11
   1a43d:	4d 01 c8             	add    %r9,%r8
   1a440:	4c 01 d9             	add    %r11,%rcx
   1a443:	f3 4e 0f b8 4c 97 10 	popcnt 0x10(%rdi,%r10,4),%r9
   1a44a:	4c 01 c8             	add    %r9,%rax
   1a44d:	f3 4e 0f b8 4c 97 18 	popcnt 0x18(%rdi,%r10,4),%r9
   1a454:	4c 01 ca             	add    %r9,%rdx
   1a457:	4d 8d 4a 08          	lea    0x8(%r10),%r9
   1a45b:	49 83 c2 10          	add    $0x10,%r10
   1a45f:	49 39 f2             	cmp    %rsi,%r10
   1a462:	4d 89 ca             	mov    %r9,%r10
   1a465:	76 c9                	jbe    1a430 <hardware_popcount+0x20>
   1a467:	4d 89 ca             	mov    %r9,%r10
   1a46a:	49 29 f2             	sub    %rsi,%r10
   1a46d:	73 62                	jae    1a4d1 <hardware_popcount+0xc1>
   1a46f:	53                   	push   %rbx
   1a470:	49 89 f3             	mov    %rsi,%r11
   1a473:	49 83 e3 03          	and    $0x3,%r11
   1a477:	74 18                	je     1a491 <hardware_popcount+0x81>
   1a479:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
   1a480:	f3 42 0f b8 1c 8f    	popcnt (%rdi,%r9,4),%ebx
   1a486:	49 01 d8             	add    %rbx,%r8
   1a489:	49 ff c1             	inc    %r9
   1a48c:	49 ff cb             	dec    %r11
   1a48f:	75 ef                	jne    1a480 <hardware_popcount+0x70>
   1a491:	49 83 fa fc          	cmp    $0xfffffffffffffffc,%r10
   1a495:	77 39                	ja     1a4d0 <hardware_popcount+0xc0>
   1a497:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
   1a49e:	00 00 
   1a4a0:	f3 46 0f b8 14 8f    	popcnt (%rdi,%r9,4),%r10d
   1a4a6:	f3 46 0f b8 5c 8f 04 	popcnt 0x4(%rdi,%r9,4),%r11d
   1a4ad:	f3 42 0f b8 5c 8f 08 	popcnt 0x8(%rdi,%r9,4),%ebx
   1a4b4:	4d 01 c2             	add    %r8,%r10
   1a4b7:	4c 01 db             	add    %r11,%rbx
   1a4ba:	f3 46 0f b8 44 8f 0c 	popcnt 0xc(%rdi,%r9,4),%r8d
   1a4c1:	4c 01 d3             	add    %r10,%rbx
   1a4c4:	49 01 d8             	add    %rbx,%r8
   1a4c7:	49 83 c1 04          	add    $0x4,%r9
   1a4cb:	4c 39 ce             	cmp    %r9,%rsi
   1a4ce:	75 d0                	jne    1a4a0 <hardware_popcount+0x90>
   1a4d0:	5b                   	pop    %rbx
   1a4d1:	48 01 d0             	add    %rdx,%rax
   1a4d4:	48 01 c8             	add    %rcx,%rax
   1a4d7:	4c 01 c0             	add    %r8,%rax
   1a4da:	c3                   	ret

Disassembly of section .init:

Disassembly of section .fini:

Disassembly of section .plt:
