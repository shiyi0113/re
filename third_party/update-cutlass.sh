set -xe

# cutlass tag
TAG_NAME="v3.9.2" 

rm -rf cutlass.origin

git clone --branch "$TAG_NAME" --single-branch https://github.com/NVIDIA/cutlass.git cutlass.origin

rm -rf cutlass
mkdir -p cutlass/tools
cp -r cutlass.origin/include cutlass
cp -r cutlass.origin/tools/util/include cutlass/tools
echo "CUTLASS version: $TAG_NAME" > cutlass/readme
git -C cutlass.origin/ rev-parse "$TAG_NAME" >> cutlass/readme 

rm -rf cutlass.origin