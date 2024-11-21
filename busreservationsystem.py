class Bus:
    def __init__(self, total_seats):
        self.total_seats = total_seats
        self.available_seats = total_seats
        self.seats = [False] * total_seats 

    def reserve_seat(self, seat_number):
        if seat_number < 1 or seat_number > self.total_seats:
            print("Invalid seat number.")
            return
        if self.seats[seat_number - 1]:
            print("Seat already reserved.")
        else:
            self.seats[seat_number - 1] = True
            self.available_seats -= 1
            print(f"Seat {seat_number} reserved successfully.")

    def cancel_reservation(self, seat_number):
        if seat_number < 1 or seat_number > self.total_seats:
            print("Invalid seat number.")
            return
        if not self.seats[seat_number - 1]:
            print("Seat is not reserved.")
        else:
            self.seats[seat_number - 1] = False
            self.available_seats += 1
            print(f"Reservation for seat {seat_number} cancelled.")

    def display_available_seats(self):
        print(f"Available seats: {self.available_seats}/{self.total_seats}")
        for i, seat in enumerate(self.seats):
            if not seat:
                print(f"Seat {i + 1} is available.")

bus = Bus(10)
bus.reserve_seat(5)
bus.reserve_seat(8)
bus.display_available_seats()
bus.cancel_reservation(5)
bus.display_available_seats()
