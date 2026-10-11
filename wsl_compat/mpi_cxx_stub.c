/* Stub for OpenMPI 4 C++ bindings (libmpi_cxx.so.40), absent in OpenMPI 5 / Ubuntu 26.04.
   torch references these only from the unused MPI process-group backend. */
void mpi_win_free(void *self) __asm__("_ZN3MPI3Win4FreeEv");
void mpi_win_free(void *self) { (void)self; }
void mpi_comm_ctor(void *self) __asm__("_ZN3MPI4CommC2Ev");
void mpi_comm_ctor(void *self) { (void)self; }
void mpi_datatype_free(void *self) __asm__("_ZN3MPI8Datatype4FreeEv");
void mpi_datatype_free(void *self) { (void)self; }
void ompi_mpi_cxx_op_intercept(void *a, void *b, int *c, void *d, void *e) { (void)a;(void)b;(void)c;(void)d;(void)e; }
void ompi_op_set_cxx_callback(void) {}
