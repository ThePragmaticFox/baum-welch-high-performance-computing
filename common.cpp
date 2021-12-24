#include "common.h"
#include "helper_utilities.h"

std::vector<struct RegisteredFunction> *FuncRegister::funcs = NULL;
std::string FuncRegister::baseline_name = "";
compute_bw_func FuncRegister::baseline_func = NULL;

void FuncRegister::add_function(compute_bw_func f, const std::string& name, const std::string& description,bool transpose_emit_prob){
    if(!funcs)
        funcs = new std::vector<struct RegisteredFunction>();

    funcs->push_back({f, name, description, transpose_emit_prob});
}

void FuncRegister::set_baseline(compute_bw_func f, const std::string& name){
    baseline_func = f;
    baseline_name = name;
}

void FuncRegister::printRegisteredFuncs(){
    printf("Registered baseline is '%s'\n", baseline_name.c_str());
    printf("User functions:\n");
    for(size_t i = 0; i < size(); i++){
        printf("%20s: %s\n", funcs->at(i).name.c_str(), funcs->at(i).description.c_str());
    }
}