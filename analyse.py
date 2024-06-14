import analysis.validity_check as vc
import analysis.instance_count as ic
import analysis.experiment_a as expt_a
import analysis.experiment_b as expt_b
import analysis.experiment_c as expt_c


def main():
    print("Validity checks...")
    vc.main()
    print("Complete")
    print("*****")

    print("Instance count...")
    ic.main()
    print("Complete")
    print("*****")

    print("Expt A - all averages")
    expt_a.a_all_averages()
    print("Complete")

    print("Expt B - all averages")
    expt_b.b_all_averages()
    print("Complete")
    print("Expt B - tables")
    for size in [10, 20, 50, 100]:
        print(size)
        expt_b.size_table(size)
    print("Complete")
    print("*****")

    print("Expt C - all averages")
    expt_c.c_all_averages()
    print("Complete")


if __name__ == "__main__":
    main()
