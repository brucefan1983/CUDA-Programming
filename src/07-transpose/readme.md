# Chapter 7: using shared memory: matrix transpose

## Source files for the chapter

| file                                 | what to learn? |
|--------------------------------------|:---------------|
| copy.cu                              | get the effective bandwidth for matrix copying |
| transpose1global_coalesced_read.cu   | coalesced read but non-coalesced write |
| transpose2global_coalesced_write.cu  | coalesced write but non-coalesced read |
| transpose3global_ldg.cu              | using `__ldg` for non-coalesced read (not needed for Pascal) |
| transpose4shared_with_conflict.cu    | using shared memory but with bank conflict |
| transpose5shared_without_conflict.cu | using shared memory and without bank conflict |


