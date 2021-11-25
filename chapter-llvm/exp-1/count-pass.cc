#include <llvm/IR/Function.h>
#include <llvm/ADT/Statistic.h>
#include <llvm/Pass.h>
#include <llvm/Support/raw_ostream.h>

using namespace llvm;

#define DEBUG_TYPE "count"

// llvm::Statistic HelloCounter = {"Hello"};
STATISTIC(HelloCounter, "Count of function call");

namespace {
    class Hello : public FunctionPass {
        public:
        static char ID;
        Hello() : FunctionPass(ID) {}

        bool runOnFunction(Function &F) override {
            ++HelloCounter;
            errs() << "Hello: ";
            errs().write_escaped(F.getName()) << "\n";
            return false;
        }
    };
}

char Hello::ID = 1;
static RegisterPass<Hello> X("count", "Count Pass");
