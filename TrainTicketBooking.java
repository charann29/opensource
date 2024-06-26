import java.awt.*;
import java.awt.event.*;
import java.util.ArrayList;
import java.util.List;

class TrainTicket {
    private String passengerName;
    private int age;
    private String source;
    private String destination;
    private int seatNumber;

    public TrainTicket(String passengerName, int age, String source, String destination, int seatNumber) {
        this.passengerName = passengerName;
        this.age = age;
        this.source = source;
        this.destination = destination;
        this.seatNumber = seatNumber;
    }

    public Object[] toArray() {
        return new Object[]{passengerName, age, source, destination, seatNumber};
    }
}

public class TrainTicketBookingAWT extends Frame implements ActionListener {
    private TextField sourceField, destField, nameField, ageField, seatField;
    private Button bookButton;
    private List<TrainTicket> tickets;
    private List<String> ticketStrings;

    public TrainTicketBookingAWT() {
        setTitle("Train Ticket Booking System");
        setSize(600, 400);
        setLocationRelativeTo(null);
        setLayout(new BorderLayout());

        tickets = new ArrayList<>();
        ticketStrings = new ArrayList<>();

        Panel inputPanel = new Panel(new GridLayout(6, 2, 5, 5));
        inputPanel.setBorder(BorderFactory.createEmptyBorder(20, 20, 20, 20));

        inputPanel.add(new Label("Source:"));
        sourceField = new TextField();
        inputPanel.add(sourceField);

        inputPanel.add(new Label("Destination:"));
        destField = new TextField();
        inputPanel.add(destField);

        inputPanel.add(new Label("Passenger Name:"));
        nameField = new TextField();
        inputPanel.add(nameField);

        inputPanel.add(new Label("Age:"));
        ageField = new TextField();
        inputPanel.add(ageField);

        inputPanel.add(new Label("Preferred Seat Number:"));
        seatField = new TextField();
        inputPanel.add(seatField);

        bookButton = new Button("Book Ticket");
        bookButton.addActionListener(this);
        inputPanel.add(bookButton);

        add(inputPanel, BorderLayout.NORTH);

        List list = new List();
        add(list, BorderLayout.CENTER);

        addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent windowEvent) {
                dispose();
            }
        });
    }

    public void actionPerformed(ActionEvent e) {
        String source = sourceField.getText();
        String destination = destField.getText();
        String name = nameField.getText();
        int age = Integer.parseInt(ageField.getText());
        int seatNumber = Integer.parseInt(seatField.getText());

        TrainTicket ticket = new TrainTicket(name, age, source, destination, seatNumber);
        tickets.add(ticket);
        ticketStrings.add(ticket.getPassengerName() + " - Seat " + ticket.getSeatNumber());

        updateList();
        clearInputFields();
    }

    private void updateList() {
        List list = (List) getComponent(1);
        list.removeAll();
        for (String ticketString : ticketStrings) {
            list.add(ticketString);
        }
    }

    private void clearInputFields() {
        sourceField.setText("");
        destField.setText("");
        nameField.setText("");
        ageField.setText("");
        seatField.setText("");
    }

    public static void main(String[] args) {
        TrainTicketBookingAWT app = new TrainTicketBookingAWT();
        app.setVisible(true);
    }
}
