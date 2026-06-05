export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Silence PSM3 / OFI libfabric NIC probe warnings on hosts without a usable
# high-perf fabric. Force Open MPI onto the safe shared-memory + TCP path.
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_mtl=^ofi,psm3
export PSM3_DEVICES=self,shm

module load mpi/openmpi-x86_64
