import analysis.instance_count as ic
import analysis.experiment_b as expt_b
import analysis.validity_check as vc


def main():
    print("Validity checks...")
    vc.main()
    print("Complete")
    print("*****")

    print("Instance count...")
    ic.main()
    print("Complete")
    print("*****")

    print("Expt B - all averages")
    expt_b.b_all_averages()
    print("Complete")
    print("Expt B - tables")
    for size in [10, 20, 50, 100]:
        print(size)
        expt_b.size_table(size)
    print("Complete")
    print("*****")


if __name__ == "__main__":
    main()
