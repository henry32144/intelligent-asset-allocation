import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Button from '@material-ui/core/Button';
import TextField from '@material-ui/core/TextField';
import Dialog from '@material-ui/core/Dialog';
import DialogActions from '@material-ui/core/DialogActions';
import DialogContent from '@material-ui/core/DialogContent';
import DialogTitle from '@material-ui/core/DialogTitle';
import CircularProgress from '@material-ui/core/CircularProgress';
import Fade from '@material-ui/core/Fade';
import Typography from '@material-ui/core/Typography';


const useStyles = makeStyles((theme) => ({
  dialogInputBox: {
    margin: theme.spacing(0, 0, 2),
  },
  dialogActions: {
    display: "flex",
    margin: theme.spacing(0, 2, 2),
  },
  errorMessage: {
    color: "#f44336",
    margin: theme.spacing(0, 2, 0),
  },
  circleProgress: {
    zIndex: 999,
    position: "absolute",
    top: "50%",
    left: "50%",
    marginTop: "-24px",
    marginLeft: "-24px"
  },
}));

export default function CreatePortfolioDialog(props) {

  const classes = useStyles();

  const [loading, setLoading] = React.useState(false);
  const [nameInputIsError, setNameError] = React.useState(false);

  const [nameInputErrorMsg, setNameErrorMsg] = React.useState("");
  const [errorMsg, setErrorMsg] = React.useState("");

  const nameInput = React.useRef();


  const checkNameInputEmpty = () => {
    if (nameInput.current.value.length < 1) {
      setNameError(true);
      setNameErrorMsg("Portfolio's name cannot be empty");
    } else {
      setNameError(false);
      setNameErrorMsg("");
    }
  }

  // Do validate here when the focus of input field is out
  // Check whether the input is empty
  const nameInputOnBlur = () => {
    checkNameInputEmpty();
  };

  const createButtonOnClick = async (e) => {
    // If no error in textfield
    checkNameInputEmpty();
    setErrorMsg("");
    if (nameInput.current.value.length > 0) {
      props.createNewPortfolio(nameInput.current.value);
    }
  };

  return (
    <div>
      <Dialog open={props.isOpen} onClose={props.handleClose} aria-labelledby="form-dialog-title">
        <Fade
          in={loading}
          unmountOnExit
        >
          <CircularProgress className={classes.circleProgress} />
        </Fade>
        <DialogTitle id="form-dialog-title">Create New Portfolio</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            inputRef={nameInput}
            margin="dense"
            id="portfolioName"
            label="Portfolio name"
            type="text"
            variant="outlined"
            fullWidth
            error={nameInputIsError}
            helperText={nameInputErrorMsg}
            className={classes.dialogInputBox}
            onBlur={nameInputOnBlur}
          />
          <Typography className={classes.errorMessage}>
            {errorMsg}
          </Typography>
        </DialogContent>
        <DialogActions className={classes.dialogActions}>
          <Button variant="contained" onClick={createButtonOnClick} color="primary">
            Create
          </Button>
        </DialogActions>
      </Dialog>
    </div>
  );
}