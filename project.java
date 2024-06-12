import java.util.ArrayList;
import java.util.Scanner;
import java.time.LocalDate;

class PassportApplication {
    private String fullName;
    private String dob;
    private String gender;
    private String nationality;
    private String phoneNumber;
    private String passportNumber;
    private String status;
    private String renewalDate;
    private String expirationDate;

    public PassportApplication(String fullName, String dob, String gender, String nationality, String phoneNumber, String passportNumber, String status) {
        this.fullName = fullName;
        this.dob = dob;
        this.gender = gender;
        this.nationality = nationality;
        this.phoneNumber = phoneNumber;
        this.passportNumber = passportNumber;
        this.status = status;
        this.renewalDate = "N/A";
        this.expirationDate = LocalDate.now().plusYears(10).toString(); // Default expiration date is 10 years from now
    }

    public String getPassportNumber() {
        return passportNumber;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public void setRenewalDate(String renewalDate) {
        this.renewalDate = renewalDate;
    }

    public String getExpirationDate() {
        return expirationDate;
    }

    public void setExpirationDate(String expirationDate) {
        this.expirationDate = expirationDate;
    }

    public void setPhoneNumber(String phoneNumber) {
        this.phoneNumber = phoneNumber;
    }

    public void setNationality(String nationality) {
        this.nationality = nationality;
    }

    @Override
    public String toString() {
        return "Full Name: " + fullName + ", DOB: " + dob + ", Gender: " + gender + ", Nationality: " + nationality + ", Phone Number: " + phoneNumber + ", Passport Number: " + passportNumber + ", Status: " + status + ", Renewal Date: " + renewalDate + ", Expiration Date: " + expirationDate;
    }
}

public class PassportAutomationSystem {
    private static ArrayList<PassportApplication> applications = new ArrayList<>();

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int choice;

        do {
            System.out.println("\nPassport Automation System");
            System.out.println("1. Apply for Passport");
            System.out.println("2. Renew Passport");
            System.out.println("3. Check Application Status");
            System.out.println("4. Update Personal Details");
            System.out.println("5. Cancel Passport Application");
            System.out.println("6. View All Applications");
            System.out.println("7. Exit");
            System.out.print("Enter your choice: ");
            choice = scanner.nextInt();
            scanner.nextLine();  // Consume newline

            switch (choice) {
                case 1:
                    applyForPassport(scanner);
                    break;
                case 2:
                    renewPassport(scanner);
                    break;
                case 3:
                    checkApplicationStatus(scanner);
                    break;
                case 4:
                    updatePersonalDetails(scanner);
                    break;
                case 5:
                    cancelPassportApplication(scanner);
                    break;
                case 6:
                    viewAllApplications();
                    break;
                case 7:
                    System.out.println("Exiting...");
                    break;
                default:
                    System.out.println("Invalid choice. Please try again.");
            }
        } while (choice != 7);
    }

    private static void applyForPassport(Scanner scanner) {
        System.out.print("Enter your full name: ");
        String fullName = scanner.nextLine();
        System.out.print("Enter your date of birth (YYYY-MM-DD): ");
        String dob = scanner.nextLine();
        System.out.print("Enter your gender (Male/Female/Other): ");
        String gender = scanner.nextLine();
        System.out.print("Enter your nationality: ");
        String nationality = scanner.nextLine();
        System.out.print("Enter your phone number: ");
        String phoneNumber = scanner.nextLine();
        String passportNumber = "P" + (applications.size() + 1);
        String status = "Submitted";

        PassportApplication application = new PassportApplication(fullName, dob, gender, nationality, phoneNumber, passportNumber, status);
        applications.add(application);
        System.out.println("Passport application submitted. Your passport number is " + passportNumber);
    }

    private static void renewPassport(Scanner scanner) {
        System.out.print("Enter your passport number: ");
        String passportNumber = scanner.nextLine();

        for (PassportApplication application : applications) {
            if (application.getPassportNumber().equals(passportNumber)) {
                application.setStatus("Renewed");
                application.setRenewalDate(LocalDate.now().toString());
                application.setExpirationDate(LocalDate.now().plusYears(10).toString());
                System.out.println("Passport renewed successfully.");
                return;
            }
        }
        System.out.println("Passport not found.");
    }

    private static void checkApplicationStatus(Scanner scanner) {
        System.out.print("Enter your passport number: ");
        String passportNumber = scanner.nextLine();

        for (PassportApplication application : applications) {
            if (application.getPassportNumber().equals(passportNumber)) {
                System.out.println(application);
                return;
            }
        }
        System.out.println("Passport not found.");
    }

    private static void updatePersonalDetails(Scanner scanner) {
        System.out.print("Enter your passport number: ");
        String passportNumber = scanner.nextLine();

        for (PassportApplication application : applications) {
            if (application.getPassportNumber().equals(passportNumber)) {
                System.out.print("Enter your new phone number: ");
                String phoneNumber = scanner.nextLine();
                System.out.print("Enter your new nationality: ");
                String nationality = scanner.nextLine();
                application.setPhoneNumber(phoneNumber);
                application.setNationality(nationality);
                System.out.println("Personal details updated successfully.");
                return;
            }
        }
        System.out.println("Passport not found.");
    }

    private static void cancelPassportApplication(Scanner scanner) {
        System.out.print("Enter your passport number: ");
        String passportNumber = scanner.nextLine();

        for (PassportApplication application : applications) {
            if (application.getPassportNumber().equals(passportNumber)) {
                applications.remove(application);
                System.out.println("Passport application cancelled successfully.");
                return;
            }
        }
        System.out.println("Passport not found.");
    }

    private static void viewAllApplications() {
        if (applications.isEmpty()) {
            System.out.println("No applications found.");
        } else {
            for (PassportApplication application : applications) {
                System.out.println(application);
            }
        }
    }
}
